import numpy as np
import torch
import random
from typing import List, Optional, Union


class MultiEnvReplayBuffer(object):
    """Buffer to store environment transitions for multiple environments"""

    def __init__(
        self, obs_shape, action_shape, capacity, device, num_envs: int, seq_len: int
    ):
        self.env_id_to_replay_buffer_map = [
            ReplayBuffer(
                obs_shape=obs_shape,
                action_shape=action_shape,
                capacity=int(capacity / num_envs),
                device=device, seq_len=seq_len
            )
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs
        self.seq_len = seq_len

    def add(self, env_id, obs, action, reward, next_obs, done, done_no_max):
        self.env_id_to_replay_buffer_map[env_id].add(
            obs, action, reward, next_obs, done, done_no_max
        )

    def add_loop(self, obs, action, reward, next_obs, done):
        for env_id in range(self.num_envs):
            self.env_id_to_replay_buffer_map[env_id].add(
                obs=obs[env_id],
                action=action[env_id],
                reward=reward[env_id],
                next_obs=next_obs[env_id],
                done=done[env_id],
            )

    def sample(self, batch_size, env_id: Optional[int] = None):
        if env_id is None:
            env_id = random.randint(0, self.num_envs - 1)
        return self.env_id_to_replay_buffer_map[env_id].sample(batch_size)

    def save(self, save_dir):
        for replay_buffer in self.env_id_to_replay_buffer_map:
            replay_buffer.save(save_dir)

    def load(self, save_dir):
        for replay_buffer in self.env_id_to_replay_buffer_map:
            replay_buffer.load(save_dir)

    def __len__(self):
        return self.env_id_to_replay_buffer_map[0].idx if not self.env_id_to_replay_buffer_map[0].full else capacity


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, seq_len=1):
        self.capacity = capacity
        self.device = device
        self.seq_len = seq_len

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.valid_inds = []

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        if done:  #  go back seq_len 
            for i in range(self.seq_len):
                if self.idx - i in self.valid_inds:
                    self.valid_inds.remove(self.idx - i)
        else:
            self.valid_inds.append(self.idx)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.choice(self.valid_inds,
                                 size=batch_size)

        obses = torch.as_tensor([self.obses[idxs+i] for i in range(self.seq_len)], device=self.device).float()
        actions = torch.as_tensor([self.actions[idxs+i] for i in range(self.seq_len)], device=self.device)
        rewards = torch.as_tensor([self.rewards[idxs+i] for i in range(self.seq_len)], device=self.device)
        next_obses = torch.as_tensor([self.next_obses[idxs+i] for i in range(self.seq_len)],
                                     device=self.device).float()
        not_dones = torch.as_tensor([self.not_dones[idxs+i] for i in range(self.seq_len)], device=self.device)
        not_dones_no_max = torch.as_tensor([self.not_dones_no_max[idxs] for i in range(self.seq_len)],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max