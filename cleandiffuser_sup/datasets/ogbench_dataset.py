from typing import Dict

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply

# Adapted from: https://github.com/leekwoon/scots/blob/main/cleandiffuser_ex/dataset/ogbench_dataset.py
class OGBenchDataset(BaseDataset):

    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            max_path_length: int = 4001,
    ):
        super().__init__()

        observations, actions, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["terminals"])
        
        dones = terminals
        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_act = [], []

        self.path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(terminals.shape[0]):

            if i != 0 and ((dones[i - 1] and not dones[i])): # finished on (i-1)-th step, new episode on i-th step

                path_length = i - ptr
                self.path_lengths.append(path_length)

                # 1. agent reaches the goal early
                if path_length < max_path_length:

                    _seq_obs = np.zeros((max_path_length, self.o_dim), dtype=np.float32)
                    _seq_act = np.zeros((max_path_length, self.a_dim), dtype=np.float32)

                    _seq_obs[:i - ptr] = normed_observations[ptr:i]
                    _seq_act[:i - ptr] = actions[ptr:i]

                    # repeat padding
                    _seq_obs[i - ptr:] = normed_observations[i]  # repeat last state
                    _seq_act[i - ptr:] = 0  # repeat zero action

                    self.seq_obs.append(_seq_obs)
                    self.seq_act.append(_seq_act)

                # 2. agent reaches the goal on time
                elif path_length == max_path_length:

                    self.seq_obs.append(normed_observations[ptr:i])
                    self.seq_act.append(actions[ptr:i])

                else:
                    raise ValueError(f"path_length: {path_length} > max_path_length: {max_path_length}")

                max_start = min(self.path_lengths[-1] - 1 - horizon, max_path_length - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i
                path_idx += 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data

# Adapted from: https://github.com/seohongpark/ogbench/blob/21cbbb084694bdac69087eed75ec6e99102ac610/impls/utils/datasets.py
class GCDataset(BaseDataset):
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    def __init__(self, dataset, config, normalizer, preprocess_frame_stack=True):
        self.dataset = dataset
        self.config = config
        self.normalizer = normalizer
        self.preprocess_frame_stack = preprocess_frame_stack
        self.size = len(self.dataset['observations'])

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        # if self.config['frame_stack'] is not None:
        #     # Only support compact (observation-only) datasets.
        #     assert 'next_observations' not in self.dataset
        #     if self.preprocess_frame_stack:
        #         stacked_observations = self.get_stacked_observations(np.arange(self.size))
        #         self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def __getitem__(self, idx):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        batch = {}
        for k, v in self.dataset.items():
            batch[k] = v[idx]

        batch['observations'] = self.get_observations(idx)
        batch['next_observations'] = self.get_observations(np.minimum(idx + 1, self.size - 1))

        value_goal_idxs = self.sample_goals(
            idx,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idx,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idx == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # if self.config['p_aug'] is not None and not evaluation:
        #     if np.random.rand() < self.config['p_aug']:
        #         self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return dict_apply(batch, torch.tensor)

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs) if type(idxs) is list or type(idxs) is np.ndarray else 1

        # Random goals.
        random_goal_idxs = np.random.randint(self.size, size=batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6), traj_goal_idxs, random_goal_idxs
        )

        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs.squeeze()

    # def augment(self, batch, keys):
    #     """Apply image augmentation to the given keys."""
    #     padding = 3
    #     batch_size = len(batch[keys[0]])
    #     crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
    #     crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
    #     for key in keys:
    #         batch[key] = jax.tree_util.tree_map(
    #             lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
    #             batch[key],
    #         )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return  self.normalizer.normalize(self.dataset['observations'][idxs])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(self.normalizer.normalize(self.dataset['observations'][cur_idxs]))
        return  np.concatenate(rets, axis=-1)
    
    def __len__(self):
        return self.size