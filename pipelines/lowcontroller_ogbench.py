import os
import ogbench
import hydra, wandb, uuid
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import trange

import torch
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import set_seed
from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset, GCDataset
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from evaluate import low_controller_evaluate
from pipelines.utils import get_wandb_video


@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):
    """Evaluate only the low-level controller on OGBench tasks.

    This ignores the high-level diffusion planner and directly lets
    the low_controller act using (obs, goal) pairs.
    """

    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.run_alias) + "_lowctrl_" + str(args.mode),
            config=OmegaConf.to_container(args, resolve=True),
        )

    set_seed(args.seed)

    save_path = f"results/{args.pipeline_name}/{args.task.env_name}_LOW/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ---------------------- Create Env & Dataset ----------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        args.task.env_name,
        compact_dataset=True,
    )
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # For evaluation, we still need a normalizer, so reuse OGBenchDataset
    planner_dataset = OGBenchDataset(
        dataset,
        horizon=args.task.planner_horizon,
        max_path_length=args.task.max_path_length,
    )

    # Policy dataset for GCIQL training (goal-conditioned)
    policy_dataset = GCDataset(
        dataset,
        args.low_controller,
        planner_dataset.get_normalizer(),
        preprocess_frame_stack=False,
    )
    policy_dataloader = DataLoader(
        policy_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---------------------- Build Low Controller (GCIQL) ----------------------
    invdyn = GCIQLAgent(
        obs_dim=obs_dim,
        action_dim=act_dim,
        goal_dim=obs_dim,
        config=args.low_controller,
        device=args.device,
    )

    # ---------------------- Training ----------------------
    if args.mode == "train":
        invdyn.train()
        n_gradient_step = 0
        log = {
            "gradient_steps": 0,
            "policy_loss_value": 0.0,
            "policy_loss_critic": 0.0,
            "policy_loss_actor": 0.0,
            "q_loss": 0.0,
            "bc_loss": 0.0,
            "bc_log_prob": 0.0,
        }

        from tqdm import tqdm

        pbar = tqdm(total=args.invdyn_gradient_steps / args.log_interval)

        for batch in loop_dataloader(policy_dataloader):
            info = invdyn.update(batch)
            log["policy_loss_value"] += info["value/value_loss"]
            log["policy_loss_critic"] += info["critic/critic_loss"]
            log["policy_loss_actor"] += info["actor/actor_loss"]
            log["q_loss"] += info["actor/q_loss"]
            log["bc_loss"] += info["actor/bc_loss"]
            log["bc_log_prob"] += info["actor/bc_log_prob"]

            n_gradient_step += 1

            if n_gradient_step % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step
                for k in log.keys():
                    if k != "gradient_steps":
                        log[k] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step)
                pbar.update(1)
                log = {
                    "gradient_steps": 0,
                    "policy_loss_value": 0.0,
                    "policy_loss_critic": 0.0,
                    "policy_loss_actor": 0.0,
                    "q_loss": 0.0,
                    "bc_loss": 0.0,
                    "bc_log_prob": 0.0,
                }
            
            # evaluate
            if n_gradient_step % args.eval_interval == 0:
                invdyn.eval()
                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
                num_tasks = len(task_infos)

                for task_id in trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]["task_name"]
                    eval_info, trajs, cur_renders = low_controller_evaluate(
                        low_controller=invdyn,
                        env=env,
                        normalizer=planner_dataset.get_normalizer(),
                        task_id=task_id,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        config=args,
                        num_eval_episodes=args.num_eval_episodes,
                        num_video_episodes=args.num_video_episodes,
                        video_frame_skip=args.video_frame_skip,
                    )

                    renders.extend(cur_renders)
                    metric_names = ["success"]
                    eval_metrics.update(
                        {f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)

                for k, v in overall_metrics.items():
                    eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

                if args.num_video_episodes > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video 

                if args.enable_wandb:
                    wandb.log(eval_metrics, step=n_gradient_step)
                
                invdyn.train()

            if n_gradient_step % args.save_interval == 0:
                invdyn.save(os.path.join(save_path, f"{args.run_alias}_lowctrl_ckpt_{n_gradient_step}.pt"))
                invdyn.save(os.path.join(save_path, f"{args.run_alias}_lowctrl_ckpt_latest.pt"))

            if n_gradient_step >= args.invdyn_gradient_steps:
                break
        
    # ---------------------- Inference / Evaluation ----------------------
    elif args.mode == "inference":
        invdyn.load(os.path.join(save_path, f"{args.run_alias}_lowctrl_ckpt_{args.invdyn_ckpt}.pt"))
        invdyn.eval()

        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
        num_tasks = len(task_infos)

        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]["task_name"]
            eval_info, trajs, cur_renders = low_controller_evaluate(
                low_controller=invdyn,
                env=env,
                normalizer=planner_dataset.get_normalizer(),
                task_id=task_id,
                obs_dim=obs_dim,
                act_dim=act_dim,
                config=args,
                num_eval_episodes=args.num_eval_episodes,
                num_video_episodes=args.num_video_episodes,
                video_frame_skip=args.video_frame_skip,
            )

            renders.extend(cur_renders)
            metric_names = ["success"]
            eval_metrics.update(
                {f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

        if args.num_video_episodes > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics['video'] = video 

        if args.enable_wandb:
            wandb.log(eval_metrics, step=1)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
