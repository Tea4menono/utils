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
        self.total_power = 20
        self.total_bandwidth = 1
        self.max_count = 0

        self.user_positions = []
        for _ in range(self.user_number):
            r = self.radius * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.user_positions.append((x, y))

        # 20 users
        self.user_positions = np.array([(151.3120098584363, 90.17385922336058),
                                        (19.721418451387617, -192.0168134106971),
                                        (282.997552859035, -25.117500656951336),
                                        (-77.27761526170703, -217.7746093751102),
                                        (9.838087681429377, -3.2348210157293584),
                                        (-174.5861043891205, 237.09766167611966),
                                        (-191.6366491221487, 42.41536937973886),
                                        (-75.56644503778357, 20.205776933611432),
                                        (54.6618078165047, -35.613353278283945),
                                        (-47.556017283810256, -123.86946865248639),
                                        (184.45947862240845, -132.94999779939536),
                                        (-154.6235336781708, -165.7987822588361),
                                        (227.10557041413378, 107.34779553265385),
                                        (163.88286570419424, 124.39377026403695),
                                        (82.26650141424528, 231.69810088321785),
                                        (205.3032016298353, -152.04156287442294),
                                        (-123.96140858005407, -179.4790843057286),
                                        (100.92959893903434, 139.56435626221938),
                                        (-18.861422818634576, -194.75645653406048),
                                        (201.0289672730241, -206.5560652803616)])
        # uav height change +1m ~ -1m
        # power change for users +0.1w ~ -0.1w  10 users
        # bandwidth change for each users

        action_low_bounds = [-1] + [0] * \
            self.user_number + [0] * self.user_number
        action_low_bounds = np.array(action_low_bounds, dtype=np.float32)

        action_high_bounds = [1] + [0.1] * \
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

    def reset(self, seed=None, options=None):
        self.uav_height = 200

        # set equal value
        self.uav_power = [self.total_power /
                          self.user_number] * self.user_number
        self.uav_bandwidth = [1 / self.user_number] * self.user_number

        info = {}
        return self._get_obs(), info

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

        # calculate coverage percent
        coverage = count / self.user_number
        reward = coverage

        # penalty

        # use ratio
        # if np.sum(self.uav_power) - self.total_power > 0:
        #     reward -= (np.sum(self.uav_power) -
        #                self.total_power)/self.total_power

        # if np.sum(self.uav_bandwidth) - 1 > 0:
        #     reward -= (np.sum(self.uav_bandwidth) - 1)/1

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
