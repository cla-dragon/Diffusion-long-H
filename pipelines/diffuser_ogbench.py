import os
import csv
import ogbench
import gymnasium
from ogbench.utils import load_dataset
from collections import defaultdict
import hydra, wandb, uuid, tempfile
import numpy as np
from tqdm import tqdm
from tqdm import trange
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import loop_dataloader, loop_two_dataloaders
from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.diffusion import DiscreteDiffusionSDE, ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d, DiT1d, DVInvMlp
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_condition import MLPCondition, IdentityCondition
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from cleandiffuser.utils import report_parameters, set_seed

from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset, GCDataset
from evaluate import single_layer_evaluate
from pipelines.utils import get_wandb_video

@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.run_alias)+"_"+str(args.mode),
            config=OmegaConf.to_container(args, resolve=True)
        )

    set_seed(args.seed)
    # TODO: change save_path
    save_path = f'results/{args.pipeline_name}/{args.task.env_name}_H{args.task.planner_horizon}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        args.task.env_name,
        compact_dataset=True,
    )
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    planner_dataset = OGBenchDataset(
        dataset,
        horizon=args.task.planner_horizon,
        max_path_length=args.task.max_path_length,
    )
    if args.use_diffusion_invdyn:
        policy_dataset = OGBenchDataset(
            dataset,
            horizon=args.task.planner_horizon,
            max_path_length=args.task.max_path_length,
        )
    else:
        policy_dataset = GCDataset(
            dataset,
            args.low_controller,
            planner_dataset.get_normalizer(),
            preprocess_frame_stack=False,
        )
    planner_dataloader = DataLoader(
        planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    policy_dataloader = DataLoader(
        policy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # --------------- Network Architecture -----------------
    nn_diffusion_planner = DiT1d(
            obs_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//64, depth=args.planner_depth, timestep_emb_type="fourier")
    # nn_classifier = HalfJannerUNet1d(
    #     args.task.horizon, obs_dim*2 + act_dim, out_dim=1, # note: obs_dim*2 for goal-conditioned Q(s,a,g)
    #     model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
    #     timestep_emb_type="positional", kernel_size=3)
    print(f"=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    # NOTE:No classifier guidance for Goal-Conditioned OGBench for now
    # classifier = CumRewClassifier(nn_classifier, device=args.device)

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.planner_horizon, obs_dim))
    fix_mask[0, :obs_dim] = 1.
    fix_mask[-1, :obs_dim] = 1.  # condition on goal state
    loss_weight = torch.ones((args.task.planner_horizon, obs_dim))
    loss_weight[0, obs_dim:] = args.planner_next_obs_loss_weight

    # --------------- Diffusion Model --------------------
    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition=None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.planner_ema_rate,
        device=args.device, predict_noise=args.planner_predict_noise, noise_schedule="linear")
    
    # ---------------------- Inverse Dynamic (Policy) -----------------------
    if args.use_diffusion_invdyn:
        nn_diffusion_invdyn = DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=args.policy_hidden_dim, timestep_emb_type="positional").to(args.device)
        nn_condition_invdyn = IdentityCondition(dropout=0.0).to(args.device)
        print(f"=============== Parameter Report of Policy ===================================")
        report_parameters(nn_diffusion_invdyn)
        print(f"==============================================================================")
        # --------------- Diffusion Model Actor --------------------
        policy = DiscreteDiffusionSDE(
            nn_diffusion_invdyn, nn_condition_invdyn, predict_noise=args.policy_predict_noise, optim_params={"lr": args.policy_learning_rate},
            x_max=+1. * torch.ones((1, act_dim), device=args.device),
            x_min=-1. * torch.ones((1, act_dim), device=args.device),
            diffusion_steps=args.policy_diffusion_steps, ema_rate=args.policy_ema_rate, device=args.device)
    else:
        invdyn = GCIQLAgent(
            obs_dim=obs_dim,
            action_dim=act_dim,
            goal_dim=obs_dim,           # 这里假设 goal 也是 state 维度
            config=args.low_controller,
            device=args.device,
        )
    
    # ---------------------- Training ----------------------
    if args.mode == "train":

        planner_lr_scheduler = CosineAnnealingLR(planner.optimizer, args.planner_diffusion_gradient_steps)
        planner.train()
        # Policy
        if args.use_diffusion_invdyn:
            policy_lr_scheduler = CosineAnnealingLR(policy.optimizer, args.policy_diffusion_gradient_steps)
            policy.train()
        else:
            # invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
            #NOTE GCIQL 不考虑 lr_scheduler
            invdyn.train()

        n_gradient_step = 0
        if args.use_diffusion_invdyn:
            log = {"gradient_steps": 0, "avg_loss_planner": 0., "bc_loss_policy": 0.}
        else:
            log = {"gradient_steps": 0, "avg_loss_planner": 0., 
                   "policy_loss_value": 0., "policy_loss_critic": 0., "policy_loss_actor": 0.}
        pbar = tqdm(total=max(args.planner_diffusion_gradient_steps, args.policy_diffusion_gradient_steps)/args.log_interval)

        for planner_batch, policy_batch in loop_two_dataloaders(planner_dataloader, policy_dataloader):

            planner_horizon_obs = planner_batch["obs"]["state"].to(args.device)
            planner_horizon_action = planner_batch["act"].to(args.device)
            planner_horizon_data = planner_horizon_obs

            if args.use_diffusion_invdyn:
                policy_horizon_obs = policy_batch["obs"]["state"].to(args.device)
                policy_horizon_action = policy_batch["act"].to(args.device)
                policy_td_obs, policy_td_next_obs, policy_td_act = policy_horizon_obs[:,0,:], policy_horizon_obs[:,1,:], policy_horizon_action[:,0,:]

            # ----------- Gradient Step ------------
            log["avg_loss_planner"] += planner.update(planner_horizon_data)['loss']
            planner_lr_scheduler.step()
            if args.use_diffusion_invdyn:
                # ----------- Policy Gradient Step ------------
                if n_gradient_step <= args.policy_diffusion_gradient_steps:
                    log["bc_loss_policy"] += policy.update(policy_td_act, torch.cat([policy_td_obs, policy_td_next_obs], dim=-1))['loss']
                    policy_lr_scheduler.step()
            else:    
                if n_gradient_step <= args.invdyn_gradient_steps:
                    info = invdyn.update(policy_batch)
                    log["policy_loss_value"] += info['value/value_loss']
                    log["policy_loss_critic"] += info['critic/critic_loss']
                    log["policy_loss_actor"] += info['actor/actor_loss']
                    # invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                for key in log.keys():
                    if key != "gradient_steps":
                        log[key] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                pbar.update(1)
                if args.use_diffusion_invdyn:
                    log = {"gradient_steps": 0, "avg_loss_planner": 0., "bc_loss_policy": 0.}
                else:
                    log = {"gradient_steps": 0, "avg_loss_planner": 0., 
                        "policy_loss_value": 0., "policy_loss_critic": 0., "policy_loss_actor": 0.}
                    
            # ----------- Evalutation ------------
            if (n_gradient_step + 1) % args.eval_interval == 0:
                planner.eval()
                if args.use_diffusion_invdyn:
                    policy.eval()
                else:
                    invdyn.eval()

                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
                num_tasks = len(task_infos)
                for task_id in trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]['task_name']
                    eval_info, trajs, cur_renders = single_layer_evaluate(
                        diffusions_model=planner,
                        mode=args.low_controller_mode,
                        low_controller=policy if args.use_diffusion_invdyn else invdyn,
                        env=env,
                        normalizer=planner_dataset.get_normalizer(),
                        task_id=task_id,
                        horizon=args.task.planner_horizon,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        config=args,
                        num_eval_episodes=args.num_eval_episodes,
                        num_video_episodes=args.num_video_episodes,
                        video_frame_skip=args.video_frame_skip,
                    )
                    renders.extend(cur_renders)
                    metric_names = ['success']
                    eval_metrics.update(
                        {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)

                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                if args.num_video_episodes > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video 

                wandb.log(eval_metrics, step=n_gradient_step + 1)

                planner.train()
                if args.use_diffusion_invdyn:
                    policy.train()
                else:
                    invdyn.train()
            
            # ----------- Save Model ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_{n_gradient_step + 1}.pt"))
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_latest.pt"))
                if args.use_diffusion_invdyn:
                    policy.save(os.path.join(save_path, f"{args.run_alias}_policy_ckpt_{n_gradient_step + 1}.pt"))
                    policy.save(os.path.join(save_path, f"{args.run_alias}_policy_ckpt_latest.pt"))
                else:
                    invdyn.save(os.path.join(save_path, f"{args.run_alias}_invdyn_ckpt_{n_gradient_step + 1}.pt"))
                    invdyn.save(os.path.join(save_path, f"{args.run_alias}_invdyn_ckpt_latest.pt"))


            n_gradient_step += 1
            if n_gradient_step >= args.planner_diffusion_gradient_steps and n_gradient_step >= args.policy_diffusion_gradient_steps:
                print(f"===================== Training Finished =====================")
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        planner.load(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_{args.planner_ckpt}.pt"))
        planner.eval()
        if args.use_diffusion_invdyn:
            policy.load(os.path.join(save_path, f"{args.run_alias}_policy_ckpt_{args.policy_ckpt}.pt"))
            policy.eval()
        else:
            invdyn.load(os.path.join(save_path, f"{args.run_alias}_invdyn_ckpt_{args.invdyn_ckpt}.pt"))
            invdyn.eval()

        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = len(task_infos)
        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders = single_layer_evaluate(
                diffusions_model=planner,
                mode=args.low_controller_mode,
                low_controller=policy if args.use_diffusion_invdyn else invdyn,
                env=env,
                normalizer=planner_dataset.get_normalizer(),
                task_id=task_id,
                horizon=args.task.planner_horizon,
                obs_dim=obs_dim,
                act_dim=act_dim,
                config=args,
                num_eval_episodes=args.num_eval_episodes,
                num_video_episodes=args.num_video_episodes,
                video_frame_skip=args.video_frame_skip,
            )
            renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update(
                {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        if args.num_video_episodes > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics['video'] = video 

        wandb.log(eval_metrics, step=1)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    

if __name__ == "__main__":
    pipeline()