import os
import warnings
import numpy as np
import gym
import argparse

from reward_model import RewardModel, CNNRewardModel, TransformerRewardModel
import logger
import random
import torch
import pickle
from pathlib import Path
from cfg import parse_cfg
import glob

import d4rl


warnings.filterwarnings('ignore')

__CONFIG__, __LOGS__ = 'cfgs', 'reward_model_logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )


def qlearning_smarts_dataset(dataset=None):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        category: category name.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    N = len(dataset)
    pic_ = []
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    episode_step = 0
    for i in range(N):
        pic = dataset[i][0]
        obs = dataset[i][1].astype(np.float32)
        if i < N - 1:
            new_obs = dataset[i + 1][1].astype(np.float32)
        else:
            new_obs = obs  # place holder
        action = dataset[i][2]
        reward = dataset[i][3]
        done_bool = bool(dataset[i][4])

        if done_bool:
            episode_step = 0

        pic_.append(pic)
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

        episode_step += 1

    return {
        "pictures": None,
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def get_smarts_dataset(dataset_path=None):
    pkl_files = glob.glob(os.path.join(dataset_path, '*.pkl'))
    raw_datasets = []
    for file in pkl_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                raw_datasets.extend(data)
            else:
                print(f"The data in {file} is not a list and cannot be extended.")

    datasets = qlearning_smarts_dataset(raw_datasets)

    print("Finished, loaded {} timesteps.".format(int(datasets["rewards"].shape[0])))
    print(datasets.keys())

    return datasets


def get_atari_dataset(env):
    datasets = env.get_dataset()

    print("Finished, loaded {} timesteps.".format(int(datasets["rewards"].shape[0])))
    print(datasets.keys())

    return datasets


def load_queries_with_indices(dataset, num_query, len_query, saved_indices, saved_labels=None, label_type=1,
                              scripted_teacher=False, equivalence_threshold=0, modality="state", partition_idx=None):
    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros((num_query, len_query))

    if modality == "state":
        observation_dim = (dataset["observations"].shape[-1], )
    elif modality == "pixel":
        observation_dim = dataset["observations"].shape[-3:]
    else:
        raise ValueError("Modality error")

    action_dim = dataset["actions"].shape[-1]

    total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, len_query) + observation_dim), np.zeros(
        (num_query, len_query) + observation_dim)
    total_act_seq_1, total_act_seq_2 = np.zeros((num_query, len_query, action_dim)), np.zeros(
        (num_query, len_query, action_dim))
    total_timestep_1, total_timestep_2 = np.zeros((num_query, len_query), dtype=np.int32), np.zeros(
        (num_query, len_query), dtype=np.int32)

    if saved_labels is None:
        query_range = np.arange(num_query)
    else:
        # do not query all label
        if partition_idx is None:
            query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))
        else:
            # If dataset is large, you should load the dataset in slices.
            query_range = np.arange(partition_idx * num_query, (partition_idx + 1) * num_query)

    for query_count, i in enumerate(query_range):
        temp_count = 0
        while temp_count < 2:
            start_idx = saved_indices[temp_count][i]
            end_idx = start_idx + len_query

            reward_seq = dataset['rewards'][start_idx:end_idx]
            obs_seq = dataset['observations'][start_idx:end_idx]
            act_seq = dataset['actions'][start_idx:end_idx]
            terminal_seq = dataset['terminals'][start_idx:end_idx]

            timestep_seq = np.arange(1, len_query + 1)

            if temp_count == 0:
                total_reward_seq_1[query_count] = reward_seq
                total_obs_seq_1[query_count] = obs_seq
                total_act_seq_1[query_count] = act_seq
                total_timestep_1[query_count] = timestep_seq
            else:
                total_reward_seq_2[query_count] = reward_seq
                total_obs_seq_2[query_count] = obs_seq
                total_act_seq_2[query_count] = act_seq
                total_timestep_2[query_count] = timestep_seq

            temp_count += 1

    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()

    seg_obs_1 = total_obs_seq_1.copy()
    seg_obs_2 = total_obs_seq_2.copy()

    seq_act_1 = total_act_seq_1.copy()
    seq_act_2 = total_act_seq_2.copy()

    seq_timestep_1 = total_timestep_1.copy()
    seq_timestep_2 = total_timestep_2.copy()

    batch = {}
    # script_labels
    # label_type = 0 perfectly rational / label_type = 1 equivalence_threshold
    if label_type == 0:  # perfectly rational
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
    elif label_type == 1:
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= equivalence_threshold).reshape(-1)
        rational_labels[margin_index] = 0.5
    batch['script_labels'] = rational_labels

    if scripted_teacher:
        # fake label
        batch['labels'] = saved_labels
    else:
        # human label
        human_labels = np.zeros((len(saved_labels), 2))
        human_labels[np.array(saved_labels) == 0, 0] = 1.
        human_labels[np.array(saved_labels) == 1, 1] = 1.
        human_labels[np.array(saved_labels) == -1] = 0.5
        human_labels = human_labels[query_range]
        batch['labels'] = human_labels

    batch['observations'] = seg_obs_1
    batch['actions'] = seq_act_1
    batch['observations_2'] = seg_obs_2
    batch['actions_2'] = seq_act_2
    batch['timestep_1'] = seq_timestep_1
    batch['timestep_2'] = seq_timestep_2
    batch['start_indices'] = saved_indices[0]
    batch['start_indices_2'] = saved_indices[1]

    return batch


def train(cfg):
    # set seed
    set_seed(cfg.seed)
    # get work dir
    last_name = 'epoch_' + str(cfg.n_epochs) + '_query_' + str(cfg.num_query) +\
                '_len_' + str(cfg.len_query) + '_seed_' + str(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.env / cfg.exp_name / last_name
    print(work_dir)
    L = logger.Logger(work_dir, cfg)

    # setup environments
    if cfg.domain == "atari":
        import d4rl_atari
        gym_env = gym.make(cfg.env, stack=cfg.stack)
        dataset = get_atari_dataset(gym_env)
        # action extension
        dataset['actions'] = dataset['actions'].reshape(-1, 1)
        # transform to onehot type
        observation_dim = gym_env.observation_space.shape  # (84, 84)
        action_dim = gym_env.action_space.n  # 6
        dataset["actions"] = np.eye(action_dim)[dataset["actions"].reshape(-1)]
    elif cfg.domain in ["mujoco", "antmaze", "adroit"]:
        import d4rl
        gym_env = gym.make(cfg.env)
        dataset = get_d4rl_dataset(gym_env.unwrapped)
        dataset['actions'] = np.clip(dataset['actions'], -cfg.clip_action, cfg.clip_action)
        observation_dim = gym_env.observation_space.shape[0]
        action_dim = gym_env.action_space.shape[0]
    elif cfg.domain in ["smarts"]:
        dataset = get_smarts_dataset(cfg.raw_dataset_dir)
        # action extension
        dataset['actions'] = dataset['actions'].reshape(-1, 1)
        # transform to onehot type
        observation_dim = dataset['observations'].shape[-1]  # 235
        action_dim = max(dataset['actions'])[0] + 1  # 11
        dataset["actions"] = np.eye(action_dim)[dataset["actions"].reshape(-1)]
    else:
        raise ValueError("Domain not found!")
    print(f"Load env {cfg.env} successfully!")

    print(f"Load saved indices from {cfg.data_dir}.")

    # load human labels or fake labels
    if os.path.exists(cfg.data_dir):
        suffix = f"domain_{cfg.domain}_env_{cfg.env}_num_{cfg.num_query}_len_{cfg.len_query}"
        matched_file = []
        for file_name in os.listdir(cfg.data_dir):
            if suffix in file_name:
                print(file_name)
                matched_file.append(file_name)
        label_file, indices_1_file, indices_2_file = sorted(matched_file)

        with open(os.path.join(cfg.data_dir, label_file), "rb") as fp:  # Unpickling
            label_file = pickle.load(fp)
        with open(os.path.join(cfg.data_dir, indices_1_file), "rb") as fp:  # Unpickling
            human_indices_1 = pickle.load(fp)
        with open(os.path.join(cfg.data_dir, indices_2_file), "rb") as fp:  # Unpickling
            human_indices_2 = pickle.load(fp)
        print(label_file.shape, human_indices_1.shape, human_indices_2.shape)
    else:
        raise ValueError(f"Label not found")

    # train reward model
    if cfg.modality == "state":
        if cfg.structure == 'mlp':
            reward_model = RewardModel(cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=3e-4,
                                    activation="tanh", logger=L)
        elif "transformer" in cfg.structure:
            reward_model = TransformerRewardModel(
                cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=3e-4,
                structure_type=cfg.structure,
                d_model=cfg.d_model, num_layers=cfg.num_layers, nhead=cfg.nhead, max_seq_len=cfg.max_seq_len,
                activation="tanh", logger=L)
        pref_dataset = load_queries_with_indices(
            dataset, cfg.num_query, cfg.len_query, saved_indices=[human_indices_1, human_indices_2],
            saved_labels=label_file, scripted_teacher=cfg.fake_label, modality=cfg.modality)
        reward_model.train(n_epochs=cfg.n_epochs, pref_dataset=pref_dataset,
                           data_size=pref_dataset["observations"].shape[0],
                           batch_size=cfg.batch_size)
    else:
        reward_model = CNNRewardModel(cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=5e-4, activation=None, logger=L, device='cuda')
        N_DATASET_PARTITION = 5
        pref_dataset = [load_queries_with_indices(
            dataset, cfg.num_query//N_DATASET_PARTITION, cfg.len_query, saved_indices=[human_indices_1, human_indices_2],
            saved_labels=label_file, scripted_teacher=cfg.fake_label, modality=cfg.modality, partition_idx=p_idx)
            for p_idx in range(N_DATASET_PARTITION)]
        # data_size = None means computing data size in function.
        reward_model.split_train(n_epochs=cfg.n_epochs, pref_dataset=pref_dataset,
                                 data_size=None,
                                 batch_size=cfg.batch_size)

    L.finish(reward_model)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))

