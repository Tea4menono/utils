import numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces


# latency
# sum data rate
# reward


class UAV(gym.Env):
    def __init__(self):
        super(UAV, self).__init__()

        self.radius = 300
        self.threshold = 0.3

        self.user_number = 20
        self.total_power = 10
        self.total_bandwidth = 1
        self.max_count = 0
        self.last_reward = 0

        self.user_positions = []
        for _ in range(self.user_number):
            r = self.radius * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.user_positions.append((x, y))

        # uav height change +1m ~ -1m
        # power change for users +0.1w ~ -0.1w  10 users
        # bandwidth change for each users

        action_low_bounds = [-1] + [0] * \
            self.user_number + [0] * self.user_number
        action_low_bounds = np.array(action_low_bounds, dtype=np.float32)

        action_high_bounds = [1] + [0.01] * \
            self.user_number + [0.001] * self.user_number
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

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # Use self.np_random instead of np.random everywhere in your class
        return [seed]

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        # self.seed(seed)
        self.uav_height = 200

        # try different seeds

        # set equal value
        self.uav_power = [self.total_power /
                          self.user_number] * self.user_number

        self.uav_bandwidth = [1 / self.user_number] * self.user_number

        info = {}
        return self._get_obs(), info

    def clear_max_count(self):
        self.max_count = 0

    def step(self, action):

        power_penalty = 0
        bandwidth_penalty = 0
        self.uav_height = np.clip(self.uav_height + action[0], 200, 500)

        self.uav_power += action[1:(1 + self.user_number)]
        self.uav_bandwidth += action[(1 + self.user_number):]

        for i in range(self.user_number):
            if self.uav_power[i] < 0:
                self.uav_power[i] = 0
                self.uav_bandwidth[i] = 0
            if self.uav_bandwidth[i] < 0:
                self.uav_bandwidth[i] = 0
                self.uav_power[i] = 0

        # print(self.uav_power, "power", np.sum(self.uav_power))
        # print(self.uav_bandwidth, "bandwidth", np.sum(self.uav_bandwidth))

        if np.sum(self.uav_power) - self.total_power > 0:
            power_penalty += (np.sum(self.uav_power) - self.total_power)

        if np.sum(self.uav_bandwidth) - 1 > 0:
            bandwidth_penalty += np.sum(self.uav_bandwidth) - 1

        print(power_penalty, "power_penalty")
        print(bandwidth_penalty, "bandwidth_penalty")

        self.uav_power = self.uav_power / \
            np.sum(self.uav_power) * self.total_power
        self.uav_bandwidth = self.uav_bandwidth / \
            np.sum(self.uav_bandwidth)
        etas = []
        count = 0
        for i in range(self.user_number):
            C = 11.95
            B = 0.136
            alpha = 3
            beta = 5
            noise_dB = -120
            noise_linear = 10 ** (noise_dB / 10)
            degrees_of_freedom = 2
            e = 0.6

            h = self.uav_height
            d = np.sqrt(self.user_positions[i][0] **
                        2 + self.user_positions[i][1] ** 2 + h ** 2)

            theta = (180 / np.pi) * np.arcsin(h / d)
            probability_los = 1 / (1 + C * np.exp(-B * (theta - C)))
            probability_nlos = 1 - (1 / (1 + C * np.exp(-B * (theta - C))))

            # fading
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

        # if reward < self.last_reward:
        #     reward = reward-1
        reward -= power_penalty
        reward -= bandwidth_penalty
        self.last_reward = reward
        done = False
        if count == self.user_number:
            done = True
        truncated = False
        info = {}
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        obs = np.concatenate(
            ([self.uav_height], self.uav_power, self.uav_bandwidth))
        return obs

    def close(self):
        pass
