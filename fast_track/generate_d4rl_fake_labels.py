# -*- coding: UTF-8 -*-
"""
@Project ：Enhanced-RLHF
@File    ：generate_atari_fake_labels.py
@Author  ：Yifu Yuan
@Date    ：2023/6/30
"""
import os
import sys
import pickle
import uuid
import argparse

from tqdm import tqdm
import gym
import d4rl
import numpy as np


def load_offline_dataset(cfg, dataset_path=None):
    task = cfg["env_name"]
    category = cfg['domain']
    gym_env = gym.make(task)
    if category == 'mujoco':
        datasets = qlearning_mujoco_dataset(gym_env)
    elif category == 'adroit':
        datasets = qlearning_adroit_dataset(gym_env)
    elif category == 'antmaze':
        datasets = qlearning_ant_dataset(gym_env)
    else:
        raise ValueError(f"{category} undefined")

    print("Finished, loaded {} timesteps.".format(int(datasets["rewards"].shape[0])))
    print(datasets.keys())

    return datasets


def qlearning_mujoco_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
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
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    xy_ = []
    done_bef_ = []
    
    qpos_ = []
    qvel_ = []
    
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True
    
    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i]) or episode_step == env._max_episode_steps - 1
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)
        
        qpos = dataset["infos/qpos"][i]
        qvel = dataset["infos/qvel"][i]
        
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2
        
        done_bef = bool(next_final_timestep)
        
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
        
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        xy_.append(xy)
        done_bef_.append(done_bef)
        
        qpos_.append(qpos)
        qvel_.append(qvel)
        episode_step += 1
    
    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
        "qposes": np.array(qpos_),
        "qvels": np.array(qvel_),
    }


def qlearning_ant_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
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
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    goal_ = []
    xy_ = []
    done_bef_ = []
    
    qpos_ = []
    qvel_ = []
    
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True
    
    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i]) or episode_step == env._max_episode_steps - 1
        goal = dataset["infos/goal"][i].astype(np.float32)
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)
        
        qpos = dataset["infos/qpos"][i]
        qvel = dataset["infos/qvel"][i]
        
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2
        
        done_bef = bool(next_final_timestep)
        
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
        
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        goal_.append(goal)
        xy_.append(xy)
        done_bef_.append(done_bef)
        
        qpos_.append(qpos)
        qvel_.append(qvel)
        episode_step += 1
    
    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "goals": np.array(goal_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
        "qposes": np.array(qpos_),
        "qvels": np.array(qvel_),
    }


def qlearning_adroit_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
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
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    xy_ = []
    done_bef_ = []
    
    qpos_ = []
    qvel_ = []
    
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True
    
    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i]) or episode_step == env._max_episode_steps - 1
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)
        
        qpos = dataset["infos/qpos"][i]
        qvel = dataset["infos/qvel"][i]
        
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2
        
        done_bef = bool(next_final_timestep)
        
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
        
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        xy_.append(xy)
        done_bef_.append(done_bef)
        
        qpos_.append(qpos)
        qvel_.append(qvel)
        episode_step += 1
    
    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
        "qposes": np.array(qpos_),
        "qvels": np.array(qvel_),
    }


class DatasetSampler:
    """Specially customized sampler for d4rl"""
    
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.len_query = cfg["len_query"]
        self.num_query = cfg["num_query"]
        self.max_episode_length = cfg["max_episode_length"]
        self.dataset = kwargs["dataset"]
        self.task = cfg["env_name"]
    
    def get_episode_boundaries(self, **kwargs):
        dataset = kwargs['dataset']
        N = dataset['rewards'].shape[0]
        
        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True
        
        episode_step = 0
        start_idx, data_idx = 0, 0
        trj_idx_list = []
        for i in range(N - 1):
            if 'maze' in self.task:
                done_bool = sum(dataset['goals'][i + 1] - dataset['goals'][i]) > 0
            else:
                done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == self.max_episode_length - 1)
            if final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                trj_idx_list.append([start_idx, data_idx - 1])
                start_idx = data_idx
            if done_bool:
                episode_step = 0
                trj_idx_list.append([start_idx, data_idx])
                start_idx = data_idx + 1
            
            episode_step += 1
            data_idx += 1
        
        trj_idx_list.append([start_idx, data_idx])
        return trj_idx_list
    
    def sample(self):
        '''
            sample num_query*len_query sequences
        '''
        trj_idx_list = self.get_episode_boundaries(dataset=self.dataset)
        trj_idx_list = np.array(trj_idx_list)
        trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1  # len(trj_len_list) = dataset episode num
        # print(trj_len_list)
        
        assert max(trj_len_list) > self.len_query
        
        start_indices_1, start_indices_2 = np.zeros(self.num_query), np.zeros(self.num_query)
        end_indices_1, end_indices_2 = np.zeros(self.num_query), np.zeros(self.num_query)
        
        for query_count in range(self.num_query):
            temp_count = 0
            while temp_count < 2:
                trj_idx = np.random.choice(np.arange(len(trj_idx_list) - 1))
                len_trj = trj_len_list[trj_idx]
                
                if len_trj > self.len_query:
                    time_idx = np.random.choice(len_trj - self.len_query + 1)
                    start_idx = trj_idx_list[trj_idx][0] + time_idx
                    end_idx = start_idx + self.len_query
                    
                    assert end_idx <= trj_idx_list[trj_idx][1] + 1
                    
                    if temp_count == 0:
                        start_indices_1[query_count] = start_idx
                        end_indices_1[query_count] = end_idx
                    else:
                        start_indices_2[query_count] = start_idx
                        end_indices_2[query_count] = end_idx
                    
                    temp_count += 1
        
        start_indices_1 = np.array(start_indices_1, dtype=np.int32)  # shape: (10, )
        start_indices_2 = np.array(start_indices_2, dtype=np.int32)
        end_indices_1 = np.array(end_indices_1, dtype=np.int32)
        end_indices_2 = np.array(end_indices_2, dtype=np.int32)
        # print(start_indices_1, start_indices_2)
        return start_indices_1, start_indices_2, end_indices_1, end_indices_2
    
    
def get_fake_labels_with_indices(dataset, num_query, len_query, saved_indices, equivalence_threshold=0):
    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros((num_query, len_query))
    
    query_range = np.arange(num_query)
    for query_count, i in enumerate(tqdm(query_range, desc="get queries from saved indices")):
        temp_count = 0
        while temp_count < 2:
            start_idx = saved_indices[temp_count][i]
            end_idx = start_idx + len_query
            
            reward_seq = dataset['rewards'][start_idx:end_idx]
            
            if temp_count == 0:
                total_reward_seq_1[query_count] = reward_seq
            else:
                total_reward_seq_2[query_count] = reward_seq
                
            temp_count += 1
    
    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()
    
    batch = {}
    
    # script_labels
    sum_r_t_1 = np.sum(seg_reward_1, axis=1)
    sum_r_t_2 = np.sum(seg_reward_2, axis=1)
    binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
    rational_labels = np.zeros((len(binary_label), 2))
    rational_labels[np.arange(binary_label.size), binary_label] = 1.0
    margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= equivalence_threshold).reshape(-1)
    rational_labels[margin_index] = 0.5
    
    batch['script_labels'] = rational_labels
    batch['start_indices'] = saved_indices[0]
    batch['start_indices_2'] = saved_indices[1]
    
    return batch


# def parse_label(data_dir, domain_name, env_name, num_query, len_query, seed):
#     suffix = f"domain_{domain_name}_env_{env_name}_num_{num_query}_len_{len_query}_seed_{seed}"
#     matched_file = []
#     for file_name in os.listdir(data_dir):
#         if suffix in file_name:
#             print(file_name)
#             matched_file.append(file_name)
#     fake_label_file, indices_1_file, indices_2_file = sorted(matched_file)
#     with open(os.path.join(data_dir, fake_label_file), "rb") as fp:  # Unpickling
#         fake_label = pickle.load(fp)
#     with open(os.path.join(data_dir, indices_1_file), "rb") as fp:  # Unpickling
#         human_indices_1 = pickle.load(fp)
#     with open(os.path.join(data_dir, indices_2_file), "rb") as fp:  # Unpickling
#         human_indices_2 = pickle.load(fp)
#     print(fake_label.shape, human_indices_1.shape, human_indices_2.shape)
#     sys.exit()


def main(args):
    # some functions need dict parameters
    cfg = vars(args)
    
    np.random.seed(cfg["seed"])
    dataset = load_offline_dataset(cfg)
    sampler = DatasetSampler(cfg, dataset=dataset)
    start_indices_1, start_indices_2, end_indices_1, end_indices_2 = sampler.sample()
    
    # script_labels, start_indices, start_indices_2
    # customize equivalence threshold
    equivalence_threshold_dict = {"mujoco": 10, "antmaze": 0, "adroit": 0}
    batch = get_fake_labels_with_indices(
        dataset,
        num_query=args.num_query,
        len_query=args.len_query,
        saved_indices=[start_indices_1, start_indices_2],
        equivalence_threshold=equivalence_threshold_dict[args.domain]
    )
    print(batch)

    save_dir = os.path.join(args.save_dir, f"{args.env_name}_fake_labels")
    identifier = str(uuid.uuid4().hex)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    suffix = f"_domain_{args.domain}_env_{args.env_name}_num_{args.num_query}_len_{args.len_query}_{identifier}"

    # assert batch["start_indices"].shape[0] == batch["start_indices_2"].shape[0] == batch["script_labels"].shape[0] == \
    # args.num_query, f"{args.env_name}: {batch["start_indices"].shape[0]} / {batch["script_labels"].shape[0]}"

    print("save query indices and fake labels.")
    with open(os.path.join(save_dir,
                           "indices_1" + suffix + ".pkl"),
              "wb",
              ) as f:
        pickle.dump(batch["start_indices"], f)
    with open(os.path.join(save_dir,
                           "indices_2" + suffix + ".pkl"),
              "wb",
              ) as f:
        pickle.dump(batch["start_indices_2"], f)
    with open(os.path.join(save_dir,
                           "fake_label" + suffix + ".pkl"),
              "wb",
              ) as f:
        pickle.dump(batch["script_labels"], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--domain', type=str, default='mujoco')
    parser.add_argument('--env_name', type=str, default='pong-medium-v0', help='Environment name.')
    parser.add_argument('--save_dir', type=str, default='generated_fake_labels/', help='query path')
    parser.add_argument('--num_query', type=int, default=2000, help='number of query.')
    parser.add_argument('--len_query', type=int, default=200, help='length of each query.')
    parser.add_argument('--seed', type=int, default=777, help='seed for reproducibility.')
    parser.add_argument('--max_episode_length', type=int, default=1000, help='maximum episode length.')
    
    args = parser.parse_args()
    main(args)