import math
from collections import namedtuple
from typing import List

import gym
import numpy as np
from gym.spaces import Box, Discrete
from smarts.env.wrappers.format_obs import FormatObs
from copy import deepcopy


def make_env(env_name, train, eval_env_seed):
    available_envs = [
        "left_turn_c",
        "left_turn_t",
        "merge",
        "cruise",
        "cut_in",
        "overtake"
    ]

    if env_name in available_envs:
        if train:
            env = make(env_name, False, True, True)
            eval_env = make(env_name, False, True, train, env_seed=eval_env_seed)
        else:
            env = None
            eval_env = make(env_name, False, True, train, env_seed=eval_env_seed)
    else:
        raise NotImplementedError(f"Env {env_name} not implemented")

    return env, eval_env


def collision_forecast(vehicle_state1, vehicle_state2, l_front=0, l_back=0, w_left=0, w_right=0, steps=10):
    v1, v2 = vehicle_state1.speed, vehicle_state2.speed
    theta1, theta2 = vehicle_state1.heading + math.pi / 2, vehicle_state2.heading + math.pi / 2
    v1_vec, v2_vec = v1 * np.array([math.cos(theta1), math.sin(theta1)]), \
                     v2 * np.array([math.cos(theta2), math.sin(theta2)])
    init_pos1, init_pos2 = vehicle_state1.position[:2], vehicle_state2.position[:2]
    bound1, bound2 = vehicle_state1.bounding_box, vehicle_state2.bounding_box
    l1, w1, l2, w2 = bound1.length, bound1.width, bound2.length, bound2.width
    l2_vec = l2 / 2 * np.array([math.cos(theta2), math.sin(theta2)])
    w2_vec = w2 / 2 * np.array([math.sin(theta2), -1 * math.cos(theta2)])

    l1_front_vec, l1_back_vec = (l1 / 2 + l_front) * np.array([math.cos(theta1), math.sin(theta1)]), \
                                (l1 / 2 + l_back) * np.array([math.cos(theta1), math.sin(theta1)])
    w1_left_vec = (w1 / 2 + w_left) * np.array([math.sin(theta1), -1 * math.cos(theta1)])
    w1_right_vec = (w1 / 2 + w_right) * np.array([math.sin(theta1), -1 * math.cos(theta1)])

    for step in range(0, steps + 1, 1):
        t = 0.1 * step
        pos1, pos2 = init_pos1 + v1_vec * t, init_pos2 + v2_vec * t
        # calculate bounding points
        bps_1 = [
            pos1 + l1_front_vec - w1_left_vec,
            pos1 + l1_front_vec + w1_right_vec,
            pos1 - l1_back_vec - w1_left_vec,
            pos1 - l1_back_vec + w1_right_vec
        ]
        bps_2 = [
            pos2 + l2_vec + w2_vec,
            pos2 + l2_vec - w2_vec,
            pos2 - l2_vec + w2_vec,
            pos2 - l2_vec - w2_vec
        ]
        bps_1_front, bps1_right = bps_1[:2], [bps_1[0], bps_1[2]]

        for bp in bps_2:
            if np.dot(bp - bps_1_front[0], bps_1_front[0] - bps_1_front[1]) * \
                    np.dot(bp - bps_1_front[1], bps_1_front[0] - bps_1_front[1]) <= 0 \
                    and np.dot(bp - bps1_right[0], bps1_right[0] - bps1_right[1]) * \
                    np.dot(bp - bps1_right[1], bps1_right[0] - bps1_right[1]) <= 0:
                return step
    return steps + 1


class StuckMonitor():
    def __init__(self, history_len=5):
        self.history_len = history_len
        self.target_wp_history, self.speed_history = [], []

    @property
    def len(self):
        return len(self.target_wp_history)

    @property
    def target_wp_std(self):
        if self.len < self.history_len:
            return 0.
        else:
            return np.array(self.target_wp_history).std(0)

    @property
    def speed_norm(self):
        if self.len < self.history_len:
            return 0.
        else:
            return np.abs(np.array(self.speed_history)).mean()

    @property
    def is_stuck(self):
        if self.len < self.history_len: return False
        return np.all(self.target_wp_std < 0.5) and \
            self.speed_norm > 5.

    def push(self, target_wp, speed):
        if self.len >= self.history_len:
            self.target_wp_history = self.target_wp_history[1:]
            self.speed_history = self.speed_history[1:]
        self.target_wp_history.append(target_wp)
        self.speed_history.append(speed)


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.np_wrapper = FormatObs(env)

        self.preserved_info_single_agent = namedtuple("PreservedInfoSingleAgent", [
            'raw_obs',
            'np_obs',

            # lane index info
            'lane_index',
            'all_lane_indeces',
            'masked_all_lane_indeces',

            # road index info
            'all_road_indeces',
            'target_wps',
            'speed_limit',
            'speed',
            'is_on_goal_lane',
            'comfort',
            'classifier_input',
            'is_turnable',
            'goal_lane',
            'wrapped_obs',
            'road_all_wrong',
            'lane_on_heading',
            'distance_to_goal',
            'neighbor_collision_risk',
        ])
        self.preserved_info: dict[str, self.preserved_info_single_agent] = {}
        self.target_lane_index = {}
        self.target_road_index = {}
        self.logged_target_lane_index = None

        self.last_correct_wp_pos = {}
        self.last_correct_st = {}

        self.neighbor_info_dim = 5
        self.env_info_dim = 25
        self.all_raw_obs = None
        self.all_np_obs = None

        self.reset()

    def reset(self):
        self.target_lane_index = {}
        self.target_road_index = {}
        self.last_correct_wp_pos = {}
        self.last_correct_st = {}
        return super().reset()

    def step(self, action):
        return super().step(action)

    def cal_rel_vel(self, v1: float, theta1: float, v2: float, theta2: float) -> np.ndarray:
        ''' Calculate v1 relative to v2. '''
        return np.array([
            -np.sin(theta1) * v1 + np.sin(theta2) * v2,
            np.cos(theta1) * v1 - np.cos(theta2) * v2
        ])

    def cal_rel_heading(self, heading1: float, heading2: float) -> float:
        ''' Calculate heading1 relative to heading2. '''
        h = heading1 - heading2
        if h > np.pi: h -= np.pi
        if h < -np.pi: h += np.pi
        return h

    def cal_goal_lane(self, np_obs, raw_obs, lane_index, all_lane_indeces):
        goal_lane = np.zeros((3, 1))
        if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"): return goal_lane
        cos_thetas = np.zeros(4)
        for i in range(4):
            y1 = np_obs["waypoints"]["pos"][i, 1][:2] - np_obs["waypoints"]["pos"][i, 0][:2]
            y2 = np_obs["mission"]["goal_pos"][:2] - np_obs["waypoints"]["pos"][i, 0][:2]
            if np.linalg.norm(y2) <= 0.2 or np.linalg.norm(y1) <= 0.2: continue
            cos_thetas[i] = abs(y1 @ y2 / np.sqrt(y1 @ y1 * y2 @ y2))
        if cos_thetas.max() > 1 - 0.0001:
            l = all_lane_indeces[cos_thetas.argmax()]
            if l == -1: return goal_lane
            goal_lane[:, 0] = (l == lane_index + np.array([-1, 0, 1])).astype(np.float32)
        return goal_lane

    def get_which_road_the_lane_belongs_to(self, np_obs, mse_threshold=40, horizon=20) -> np.ndarray:
        ''' lane_indeces: [0,0,0,0] -> road_indeces: [0,1,2,-1] '''
        road_indeces = np.zeros(4)
        wp_pos = np_obs["waypoints"]["pos"][:, :horizon, :2]
        rel_wp_pos = (wp_pos - wp_pos[0, :, :])
        delta_rel_wp_pos = rel_wp_pos - rel_wp_pos[:, 0:1, :]
        mask = np.any(wp_pos, -1).reshape((4, 20, 1))
        delta_rel_wp_pos *= mask

        similarity = (delta_rel_wp_pos ** 2).sum((1, 2))
        for i in range(1, 4):
            if np.all(wp_pos[i, ...] == 0):
                road_indeces[i] = -1  # padding
                continue
            for j in range(i):
                if abs(similarity[i] - similarity[j]) < mse_threshold:
                    road_indeces[i] = road_indeces[j]  # belongs to the same road
                    break
            else:
                road_indeces[i] = road_indeces[i - 1] + 1  # belongs to another road
        return road_indeces

    def choose_target_road_index(self, np_obs, all_road_indeces) -> int:
        ''' Choose the road that is the closest to goal. '''
        wp_end_points = np_obs["waypoints"]["pos"][:, -1, :2]
        goal_point = np_obs["mission"]["goal_pos"][:2].reshape(1, -1)
        dist = ((goal_point - wp_end_points) ** 2).sum(-1)
        dist[np.where(all_road_indeces == -1)] = dist.max() + 1
        return int(all_road_indeces[dist.argmin()])

    def check_wrong_road(self, np_obs, raw_obs):
        wrong_road_indeces = np.ones(4)
        if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"): return wrong_road_indeces
        goal_point = np_obs["mission"]["goal_pos"][:2].reshape(1, -1)
        wp_pos = np_obs["waypoints"]["pos"][:, :, :2]

        for i in range(4):
            for j in range(-1, -20, -1):
                if np.any(wp_pos[i, j - 1]):
                    last_zero_index = j
                    break
            else:
                last_zero_index = j - 1
            if last_zero_index == -20:
                wrong_road_indeces[i] = 0
                continue
            wp_end_points = wp_pos[i, last_zero_index - 3:last_zero_index]
            if wp_end_points.shape[0] < 3:
                wrong_road_indeces[i] = 1
                continue
            dist_to_goal = ((goal_point - wp_end_points) ** 2).sum(-1)
            if not (dist_to_goal[0] > dist_to_goal[1] and
                    dist_to_goal[1] > dist_to_goal[2]):
                wrong_road_indeces[i] = 0
        return wrong_road_indeces

    def get_neighbor_lane_info(self, ego_state, neighbor_state):
        if ego_state.road_id == neighbor_state.road_id:
            if neighbor_state.lane_index - ego_state.lane_index < -1:
                return 0
            elif neighbor_state.lane_index - ego_state.lane_index == -1:
                return 1
            elif neighbor_state.lane_index - ego_state.lane_index == 0:
                return 2
            elif neighbor_state.lane_index - ego_state.lane_index == 1:
                return 3
            elif neighbor_state.lane_index - ego_state.lane_index > 1:
                return 4
        else:
            return -1

    def get_np_neighbor_info(self, raw_obs, pos, speed, heading, rotate_M):
        ego_state = raw_obs.ego_vehicle_state
        neighbor_pos, neighbor_speed, neighbor_heading, neighbor_lane_idx, neighbor_collision_risk = [], [], [], [], []
        for neighbor in raw_obs.neighborhood_vehicle_states:
            neighbor_pos.append(neighbor.position[:2])
            neighbor_speed.append(neighbor.speed)
            neighbor_heading.append(neighbor.heading)
            neighbor_lane_idx.append(self.get_neighbor_lane_info(ego_state, neighbor))
            collision_time = collision_forecast(ego_state, neighbor, steps=10)
            neighbor_collision_risk.append(collision_time)

        if len(neighbor_pos) != 0:
            neighbor_pos = np.concatenate(neighbor_pos).reshape(-1, 2)
            neighbor_speed = np.array(neighbor_speed)
            neighbor_heading = np.array(neighbor_heading)
            neighbor_lane_idx = np.array(neighbor_lane_idx)
            neighbor_collision_risk = np.array(neighbor_collision_risk)
            if neighbor_pos.shape[0] < 5:
                sp = 5 - neighbor_pos.shape[0]
                neighbor_pos = np.concatenate([neighbor_pos, np.ones((sp, 2)) * 200])
                neighbor_speed = np.concatenate([neighbor_speed, np.zeros(sp)])
                neighbor_heading = np.concatenate([neighbor_heading, np.zeros(sp)])
                neighbor_lane_idx = np.concatenate([neighbor_lane_idx, np.ones(sp) * -1])
                neighbor_collision_risk = np.concatenate([neighbor_collision_risk, np.ones(sp) * 11])
        else:
            neighbor_pos = np.ones((5, 2)) * 200
            neighbor_speed = np.ones(5) * speed
            neighbor_heading = np.ones(5) * heading
            neighbor_lane_idx = np.ones(5) * -1
            neighbor_collision_risk = np.ones(5) * 11

        assert neighbor_pos.shape[0] >= 5

        NeighborInfo_rel_pos = ((neighbor_pos - pos.reshape(1, 2)) @ rotate_M.T)
        rel_dist = (NeighborInfo_rel_pos ** 2).sum(-1)
        st = rel_dist.argsort()[:5]

        NeighborInfo_rel_pos = NeighborInfo_rel_pos[st]
        neighbor_heading = neighbor_heading[st]
        neighbor_speed = neighbor_speed[st]
        neighbor_lane_idx = neighbor_lane_idx[st]
        neighbor_collision_risk = neighbor_collision_risk[st]

        neighbors_rel_vel = np.empty((5, 2))
        neighbors_rel_vel[:, 0] = -np.sin(neighbor_heading) * neighbor_speed + np.sin(heading) * speed
        neighbors_rel_vel[:, 1] = np.cos(neighbor_heading) * neighbor_speed - np.cos(heading) * speed
        NeighborInfo_rel_vel = ((neighbors_rel_vel) @ rotate_M.T)

        NeighborInfo_rel_heading = (neighbor_heading - heading).reshape(5, 1)
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading > np.pi)] -= np.pi
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi

        neighbor_lane_idx = neighbor_lane_idx.reshape(5, 1)
        neighbor_collision_risk = neighbor_collision_risk.reshape(5, 1)

        return (NeighborInfo_rel_pos, NeighborInfo_rel_vel, NeighborInfo_rel_heading, neighbor_lane_idx,
                neighbor_collision_risk)

    def observation(self, all_raw_obs):
        all_np_obs = self.np_wrapper.observation(all_raw_obs)
        self.all_np_obs = all_np_obs
        self.preserved_info = dict.fromkeys(all_raw_obs.keys())
        wrapped_obs = dict.fromkeys(all_raw_obs.keys())

        for agent_id in all_raw_obs.keys():
            raw_obs, np_obs = all_raw_obs[agent_id], all_np_obs[agent_id]

            # ego_vehicle_state
            pos = np_obs["ego"]["pos"][:2]
            heading = np_obs["ego"]["heading"]
            speed = np_obs["ego"]["speed"]
            lane_index = np_obs["ego"]["lane_index"]

            jerk_linear = np.linalg.norm(np_obs["ego"]["linear_jerk"])
            jerk_angular = np.linalg.norm(np_obs["ego"]["angular_jerk"])
            comfort = jerk_linear + jerk_angular

            rotate_M = np.array([
                [np.cos(heading), np.sin(heading)],
                [-np.sin(heading), np.cos(heading)]]
            )
            if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
                distance_to_goal = -1
            else:
                distance_to_goal = np.sqrt(((raw_obs.ego_vehicle_state.mission.goal.position[:2] - pos) ** 2).sum())

            all_lane_indeces = np_obs["waypoints"]["lane_index"][:, 0]
            all_road_indeces = self.get_which_road_the_lane_belongs_to(np_obs)
            self.target_road_index[agent_id] = self.choose_target_road_index(np_obs, all_road_indeces)
            all_lane_speed_limit = np_obs["waypoints"]["speed_limit"][:, 0].reshape(4, 1)
            all_lane_width = np_obs["waypoints"]["lane_width"][:, 0].reshape(4, 1)
            all_lane_position = np_obs["waypoints"]["pos"][:, :, :2].reshape(4, 20, 2)
            all_lane_heading = np_obs["waypoints"]["heading"][:, :].reshape(4, 20)

            all_lane_rel_position = ((all_lane_position.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T).reshape(4, 20,
                                                                                                                  2)
            all_lane_rel_heading = (all_lane_heading - heading)
            all_lane_rel_heading[np.where(all_lane_rel_heading > np.pi)] -= np.pi
            all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

            # check wrong road
            wrong_road_indeces = self.check_wrong_road(np_obs, raw_obs)
            road_all_wrong = True if not np.any(wrong_road_indeces) else False
            if not road_all_wrong:
                self.last_correct_wp_pos[agent_id] = np_obs["waypoints"]["pos"][:, :, :2]

            # Env Info
            st = [0] * 3
            masked_all_lane_indeces = all_lane_indeces.copy()
            masked_all_lane_indeces[np.where(all_road_indeces != self.target_road_index[agent_id])] = -1
            if lane_index not in masked_all_lane_indeces:
                lane_index = masked_all_lane_indeces[np.where(masked_all_lane_indeces != -1)[0][0]].item()
            if agent_id not in self.target_lane_index.keys() or self.target_lane_index[
                agent_id] not in masked_all_lane_indeces:
                self.target_lane_index[agent_id] = lane_index

            EnvInfo_is_turnable = np.zeros((3, 1))
            if lane_index - 1 in masked_all_lane_indeces and lane_index > 0:
                EnvInfo_is_turnable[0] = 1.0
                st[0] = np.where(masked_all_lane_indeces == lane_index - 1)[0][0].item()
            if lane_index + 1 in masked_all_lane_indeces:
                EnvInfo_is_turnable[2] = 1.0
                st[2] = np.where(masked_all_lane_indeces == lane_index + 1)[0][0].item()
            EnvInfo_is_turnable[1] = 1.0
            st[1] = np.where(masked_all_lane_indeces == lane_index)[0][0].item()

            if not road_all_wrong:
                self.last_correct_st[agent_id] = st[1]

            speed_limit = all_lane_speed_limit[st[1]]
            lane_on_heading = all_lane_heading[st[1], 0]

            EnvInfo_rel_pos_heading = np.zeros((3, 60))  # 20 points * 3
            EnvInfo_speed_limit = np.zeros((3, 1))
            EnvInfo_width = np.zeros((3, 1))
            for i in range(3):
                if EnvInfo_is_turnable[i] == 0: continue
                EnvInfo_rel_pos_heading[i, :40] = all_lane_rel_position[st[i]].reshape(40, )
                EnvInfo_rel_pos_heading[i, 40:] = all_lane_rel_heading[st[i]].reshape(20, )
                EnvInfo_speed_limit[i] = all_lane_speed_limit[st[i]]
                EnvInfo_width[i] = all_lane_width[st[i]]

            EnvInfo_is_target = np.zeros((3, 1))
            self.logged_target_lane_index = deepcopy(self.target_lane_index)
            if self.target_lane_index[agent_id] < lane_index:
                EnvInfo_is_target[0] = 1.0
            elif self.target_lane_index[agent_id] > lane_index:
                EnvInfo_is_target[2] = 1.0
            else:
                EnvInfo_is_target[1] = 1.0

            EnvInfo_is_goal = self.cal_goal_lane(np_obs, raw_obs, lane_index, masked_all_lane_indeces).reshape(3, 1)
            is_on_goal_lane = EnvInfo_is_goal[1, 0]

            EnvInfo_index = np.linspace(0, 2, 3)[:, None]

            EgoInfo = np.array([
                speed, jerk_linear
            ]).astype(np.float32)  # 2

            EnvInfo = np.concatenate([
                EnvInfo_rel_pos_heading,  # c 60 = 20 * 3
                EnvInfo_speed_limit,  # c 1
                EnvInfo_width,  # c 1
                EnvInfo_is_target,  # d 1
                EnvInfo_is_goal,  # d 1
                EnvInfo_is_turnable,  # d 1
                EnvInfo_index,  # d 1
            ], -1).astype(np.float32)  # 66 * 3 = 198

            EnvInfo_classifier = EnvInfo

            # Neighbor Info

            (NeighborInfo_rel_pos, NeighborInfo_rel_vel, NeighborInfo_rel_heading, neighbor_lane_idx,
             neighbor_collision_risk) = self.get_np_neighbor_info(raw_obs, pos, speed, heading, rotate_M)

            NeighborInfo = np.concatenate([
                NeighborInfo_rel_pos,
                NeighborInfo_rel_vel,
                NeighborInfo_rel_heading,
                neighbor_lane_idx + 1,
                neighbor_collision_risk,
            ], -1).astype(np.float32)

            # preserved_info
            target_wps = None
            wrapped_obs[agent_id] = np.concatenate([
                EgoInfo.reshape(-1, ),
                NeighborInfo.reshape(-1, ),
                EnvInfo.reshape(-1, ),
            ])

            self.preserved_info[agent_id] = self.preserved_info_single_agent(
                raw_obs=raw_obs,
                lane_index=lane_index,
                all_lane_indeces=all_lane_indeces,
                target_wps=target_wps,
                speed_limit=speed_limit,
                np_obs=np_obs,
                is_on_goal_lane=is_on_goal_lane,
                speed=speed,
                comfort=comfort,
                classifier_input=np.concatenate([
                    NeighborInfo.reshape(-1, ),
                    EnvInfo_classifier.reshape(-1, ),
                ]),
                is_turnable=EnvInfo_is_turnable,
                goal_lane=EnvInfo_is_goal[1, :],
                wrapped_obs=wrapped_obs[agent_id],
                all_road_indeces=all_road_indeces,
                masked_all_lane_indeces=masked_all_lane_indeces,
                road_all_wrong=road_all_wrong,
                lane_on_heading=lane_on_heading,
                distance_to_goal=distance_to_goal,
                neighbor_collision_risk=neighbor_collision_risk,
            )
        self.all_raw_obs = deepcopy(all_raw_obs)
        return wrapped_obs


class EnvWrapper(gym.Wrapper):
    def __init__(self, env: ObsWrapper):
        super().__init__(env)

        self.stuck_monitor = dict.fromkeys(self.agents_id)

        self.last_act = dict.fromkeys(self.agents_id)
        self.last_acc = dict.fromkeys(self.agents_id)
        self.last_step_goal_distance = 1000
        self.last_step_on_goal_lane = False
        self.last_pos = None

        self.reward_log = None

        self.acc_options = [-1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2]

        self.action_space, self.observation_space = {}, {}
        for agent_id in self.agents_id:
            self.stuck_monitor[agent_id] = StuckMonitor(5)
            self.action_space[agent_id] = Discrete(len(self.acc_options) + 2)
            self.observation_space[agent_id] = Box(-np.inf, np.inf, shape=(2 + 5 * 7 + 3 * 66,))

    def reset(self):
        self.last_act = dict.fromkeys(self.agents_id)
        self.last_acc = dict.fromkeys(self.agents_id)
        return self.env.reset()

    def step(self, raw_act):
        if self.agents_id == {}:
            wrapped_act = {}
        else:
            wrapped_act = self.pack_action(raw_act)
        obs, _, done, info = self.env.step(wrapped_act)

        reward = self.pack_reward(raw_act, done)
        done = self.pack_done(done)
        info = self.pack_info(info)
        return obs, reward, done, info

    @property
    def agents_id(self):
        if self.preserved_info is not None:
            return self.preserved_info.keys()
        else:
            return {}

    def cal_distance(self, pos1, pos2):
        return np.sqrt(((pos1 - pos2) ** 2).sum())

    def pack_info(self, info):
        for agent_id in self.agents_id:
            info[agent_id]["comfort"] = self.preserved_info[agent_id].comfort
        return info

    def pack_action(self, action):
        wrapped_act = dict.fromkeys(self.agents_id)
        for agent_id in self.agents_id:

            raw_obs = self.preserved_info[agent_id].raw_obs
            all_lane_indeces = self.preserved_info[agent_id].masked_all_lane_indeces

            speed = self.preserved_info[agent_id].speed
            pos = raw_obs.ego_vehicle_state.position[:2]
            heading = raw_obs.ego_vehicle_state.heading - 0.0

            # keep_lane
            wps_act = 3
            if action[agent_id] in range(len(self.acc_options)):
                acc = self.acc_options[action[agent_id]]

            # change_lane_left
            elif len(self.acc_options) == action[agent_id]:
                acc = self.last_acc[agent_id] if self.last_acc[agent_id] is not None else 0.
                wps_act = 3
                if self.target_lane_index[agent_id] + 1 in all_lane_indeces and abs(heading - self.preserved_info[agent_id].lane_on_heading) < 0.05:
                    self.target_lane_index[agent_id] += 1
                    self.last_pos = pos[0]

            # change_lane_right
            elif len(self.acc_options) + 1 == action[agent_id]:
                acc = self.last_acc[agent_id] if self.last_acc[agent_id] is not None else 0.
                wps_act = 3
                if self.target_lane_index[agent_id] > 0 and self.target_lane_index[
                    agent_id] - 1 in all_lane_indeces and abs(
                        heading - self.preserved_info[agent_id].lane_on_heading) < 0.05:
                    self.target_lane_index[agent_id] -= 1
                    self.last_pos = pos[0]

            self.last_act[agent_id] = action[agent_id]
            self.last_acc[agent_id] = acc

            target_wp = raw_obs.waypoint_paths[self.target_lane_index[agent_id]][:wps_act][-1].pos

            delta_pos = target_wp - pos
            delta_pos_dist = self.cal_distance(target_wp, pos)

            exp_speed = np.clip(speed + acc, 0, self.preserved_info[agent_id].speed_limit * 1.)

            exp_heading = np.arctan2(- delta_pos[0], delta_pos[1])
            hc_id = np.abs(2 * np.pi * (np.arange(3) - 1) + exp_heading - heading).argmin()
            exp_heading += (hc_id - 1) * 2 * np.pi
            heading = np.clip(
                exp_heading,
                heading - 0.1, heading + 0.1
            )
            new_pos = pos + delta_pos / delta_pos_dist * exp_speed * 0.1
            wrapped_act[agent_id] = np.concatenate([new_pos, [heading, 0.1]])

            self.stuck_monitor[agent_id].push(target_wp, speed)

        return wrapped_act

    def pack_reward(self, raw_act, done):
        wrapped_rew = dict.fromkeys(self.agents_id)

        for agent_id in self.agents_id:
            r_forward = self.preserved_info[agent_id].raw_obs.distance_travelled if \
                self.preserved_info[agent_id].raw_obs.ego_vehicle_state.speed > 0.01 else 0
            events = self.preserved_info[agent_id].raw_obs.events
            r_events = 0
            if events.collisions or self.stuck_monitor[agent_id].is_stuck:
                r_events -= 40
            if events.off_route:    r_events -= 6
            if events.wrong_way:    r_events -= 6
            if events.reached_goal: r_events += 20
            if events.on_shoulder:  r_events -= 6
            if events.off_road:     r_events -= 6

            r_time_cost = - 0.3

            r_lane_change = -1. if self.last_act[agent_id] >= len(self.acc_options) else 0

            r_goal = 0.3 if self.preserved_info[agent_id].is_on_goal_lane else 0.0
            if done[agent_id] and hasattr(self.preserved_info[agent_id].raw_obs.ego_vehicle_state.mission.goal,
                                          "position"):
                position = self.preserved_info[agent_id].raw_obs.ego_vehicle_state.position[:2]
                goal_position = np.array(
                    self.preserved_info[agent_id].raw_obs.ego_vehicle_state.mission.goal.position[:2])
                goal_distance = self.cal_distance(position, goal_position)
                if goal_distance < 20:
                    r_goal += min(5 / goal_distance, 10)

            r_collision_forcast = 0

            r_over_speed = 0

            wrapped_rew[agent_id] = sum([
                r_forward,
                r_events,
                r_time_cost,
                r_lane_change,
                r_goal,
                r_collision_forcast,
                r_over_speed])

            self.reward_log = [
                r_forward,
                r_events,
                r_time_cost,
                r_lane_change,
                r_goal,
                r_collision_forcast,
                r_over_speed]

        return wrapped_rew

    def pack_done(self, done):
        for agent_id in self.agents_id:
            done[agent_id] = done[agent_id] or self.stuck_monitor[agent_id].is_stuck
        return done


class SingleAgentWrapper(EnvWrapper):
    def __init__(self, env, default_agent_id="Agent_0"):
        super().__init__(env)
        self.agent_id = default_agent_id
        self.action_space = self.action_space[default_agent_id]
        self.observation_space = self.observation_space[default_agent_id]

    def reset(self):
        return self.env.reset()[self.agent_id].astype(np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step({self.agent_id: action})
        info[self.agent_id]["success_rate"] = 1.0 if self.env.preserved_info[self.agent_id].np_obs["events"][
            "reached_goal"] else 0.0
        return obs[self.agent_id].astype(np.float32), float(reward[self.agent_id]), float(done[self.agent_id]), info[
            self.agent_id]


class MultiGoalWrapper(gym.Wrapper):
    def __init__(self, envs=List[gym.Env]):
        super().__init__(envs[0])
        self.envs = envs
        self.current_env_idx = -1

    @property
    def n_envs(self):
        return len(self.envs)

    def reset(self):
        self.current_env_idx = (self.current_env_idx + 1) % self.n_envs
        self.env = self.envs[self.current_env_idx]
        return self.env.reset()


def make(
        scenario: str,
        visdom: bool = False,
        multi_goal: bool = True,
        sumo_headless: bool = False,
        env_seed: int = None,
):
    if not multi_goal:
        raw_env = gym.make(
            "smarts.env:multi-scenario-v0",
            scenario=scenario,
            visdom=visdom,
            sumo_headless=sumo_headless,
        )
        raw_env.seed(env_seed)
        env = SingleAgentWrapper(EnvWrapper(ObsWrapper(raw_env)))
    else:
        if scenario in ["left_turn_c", "left_turn_t"]:
            n_goals = 2
        elif scenario in ["cruise", "cut_in", "overtake", "merge"]:
            n_goals = 3
        else:
            raise NotImplementedError
        env = []
        for i in range(n_goals):
            raw_env = gym.make(
                        "smarts.env:multi-scenario-v0",
                        scenario=scenario + f'_{i}',
                        visdom=visdom,
                        sumo_headless=sumo_headless,
                    )
            if env_seed is not None:
                raw_env.seed(env_seed)
            else:
                raw_env.seed(0)
            env.append(
                SingleAgentWrapper(EnvWrapper(ObsWrapper(
                    raw_env
                )))
            )
        env = MultiGoalWrapper(env)
    return env
