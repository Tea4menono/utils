import gym
import numpy as np
from gym import spaces


class UAV(gym.Env):
    def __init__(self):
        super(UAV, self).__init__()

        self.radius = 300
        self.threshold = 0.3

        self.user_number = 10
        self.total_power = 10
        self.total_bandwidth = 1
        self.max_count = 0

        self.user_positions = []
        for _ in range(self.user_number):
            r = self.radius * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.user_positions.append((x, y))

        self.user_positions = np.array([[-242.72043072, -93.39132115],
                                        [151.79673523,  209.07594207],
                                        [-269.16338875,  131.00347775],
                                        [92.34309447, -133.81402296],
                                        [-139.14797933,   98.60834401],
                                        [-41.11810885,  295.79154635],
                                        [206.02394587,  -73.79802575],
                                        [-245.99279203, -107.31470621],
                                        [202.57135104,   41.01018199],
                                        [-200.3088382,   170.95196796]])

        # uav height change
        # power change for users
        # bandwidth change for each users

        action_low_bounds = [-2] + [-0.1] * \
            self.user_number + [-0.01] * self.user_number
        action_low_bounds = np.array(action_low_bounds, dtype=np.float32)

        action_high_bounds = [2] + [0.1] * \
            self.user_number + [0.01] * self.user_number
        action_high_bounds = np.array(action_high_bounds, dtype=np.float32)

        self.action_space = spaces.Box(low=action_low_bounds,
                                       high=action_high_bounds,
                                       dtype=np.float32)

        # uav height 200m ~ 400m
        # power for users 0w ~ 10w
        # bandwidth for users 0 ~ 1
        ob_low_bounds = [200] + [0] * self.user_number + [0] * self.user_number
        ob_low_bounds = np.array(ob_low_bounds, dtype=np.float32)

        ob_high_bounds = [400] + [self.total_power] * \
            self.user_number + [1] * self.user_number
        ob_high_bounds = np.array(ob_high_bounds, dtype=np.float32)

        self.observation_space = spaces.Box(low=ob_low_bounds,
                                            high=ob_high_bounds,
                                            dtype=np.float32)

    def reset(self):
        self.uav_height = 300

        # set equal value
        self.uav_power = [self.total_power /
                          self.user_number] * self.user_number
        self.uav_bandwidth = [1 / self.user_number] * self.user_number

        return self._get_obs()

    def clear_max_count(self):
        self.max_count = 0

    def step(self, action):

        self.uav_height = np.clip(self.uav_height + action[0], 200, 500)

        self.uav_power += action[1:(1 + self.user_number)]
        self.uav_bandwidth += action[(1 + self.user_number):]

        # should be more than 0
        self.uav_power = np.clip(self.uav_power, 0, None)
        self.uav_bandwidth = np.clip(self.uav_bandwidth, 0, None)

        self.uav_power = self.uav_power / \
            np.sum(self.uav_power) * self.total_power

        self.uav_bandwidth = self.uav_bandwidth / np.sum(self.uav_bandwidth)

        etas = []
        count = 0
        for i in range(self.user_number):
            C = 11.95
            B = 0.136
            alpha = 3
            beta = 5
            noise_dBm = -90
            noise_linear = 10 ** (noise_dBm / 10)
            degrees_of_freedom = 2
            e = 0.6

            h = self.uav_height
            d = np.sqrt(self.user_positions[i][0] **
                        2 + self.user_positions[i][1] ** 2 + h ** 2)

            theta = (180 / np.pi) * np.arcsin(h / d)
            probability_los = 1 / (1 + C * np.exp(-B * (theta - C)))
            probability_nlos = 1 - (1 / (1 + C * np.exp(-B * (theta - C))))

            snr_los = np.random.chisquare(
                degrees_of_freedom) * self.uav_power[i] * np.power(d, -alpha) / noise_linear
            snr_nlos = np.random.exponential(0.1) * e * \
                self.uav_power[i] * np.power(d, -alpha) / noise_linear

            eta_los = self.uav_bandwidth[i] * np.log2(1 + snr_los)
            eta_nlos = self.uav_bandwidth[i] * np.log2(1 + snr_nlos)
            eta = eta_los * probability_los + eta_nlos * probability_nlos

            if eta >= self.threshold * np.power(i + 1, 1 / beta):
                count = count + 1

            etas.append(eta)

        reward = count

        # penalty
        # if np.sum(self.uav_power) - self.total_power > 0:
        #     reward -= (np.sum(self.uav_power) - self.total_power)

        # if np.sum(self.uav_bandwidth) - 1 > 0:
        #     reward -= (np.sum(self.uav_bandwidth) - 1)

        done = False
        if count == self.user_number:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = np.concatenate(
            ([self.uav_height], self.uav_power, self.uav_bandwidth))
        return obs

    def close(self):
        pass
