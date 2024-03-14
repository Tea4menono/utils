from abc import ABC

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise

# global configs
uav_height = 200
radius = 300
threshold = 0.5
user_number = 50
total_power = 10
total_bandwidth = 1
user_positions = []

# set up logger

# set random positions

#
# for _ in range(user_number):
#     r = radius * np.sqrt(np.random.uniform(0, 1))
#     theta = np.random.uniform(0, 2 * np.pi)
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#
#     user_positions.append((x, y))
#

user_positions = [(-29.908560586105782, 47.79602366659078), (86.33033647874362, 273.3532821771217),
                  (162.20726428223097, -103.14009495968087), (294.7550433657454, 18.85480582738252),
                  (-252.5007689856233, -2.995352061560141), (-19.13407302810495, 202.37050580059406),
                  (237.92259837921193, 27.391558743686772), (15.394054576922864, 55.720250379686185),
                  (246.2169277525952, -160.49928256431508), (-75.44696060181158, 120.93934659502152),
                  (266.8410669898153, 31.63471473203044), (42.237902356254175, 255.96250167275358),
                  (-47.471261448490516, 97.13000395776899), (60.33635138404055, 249.5465178174473),
                  (-229.1135153059827, 153.01192227782195), (126.26586253603965, 107.39485318586703),
                  (19.963556198743888, -163.4232735585985), (-282.67774449981493, -50.88985458942966),
                  (-103.83429284436264, -151.5250313919736), (-114.55040718231943, -228.39310702436097),
                  (142.05427379001904, -10.171060924589266), (59.4470053493115, -110.75118792480296),
                  (269.62368844060865, -94.33399880220065), (-153.73414210622605, -188.38754215333697),
                  (16.3515185723441, 1.022110572740019), (232.03846084776566, -40.8484833946135),
                  (-261.9617455059748, -19.646584278259727), (201.49871100976083, 46.33604791180502),
                  (-100.38775290044933, -211.85353131521984), (22.148454732316846, 65.52275476579673),
                  (282.71699597672716, 16.096113967666053), (-4.523146488181627, -36.08271810062147),
                  (-140.58338786370095, -48.69805875153498), (-265.204532669083, 37.017054812961526),
                  (-75.7528674735538, -107.27364602896198), (-115.70631337706521, 92.98535763732814),
                  (-53.03801462814122, 233.42743818842075), (-93.20518155472307, -121.77826159571018),
                  (9.642504627397509, -140.05646408869535), (-191.99018447820646, 163.224435144345),
                  (228.25101395567629, -43.51967496732481), (201.75954507238598, -91.51730134932234),
                  (-111.98085610620541, 204.5059378465141), (-70.0295321923673, -226.31047722782452),
                  (111.54612629620564, 67.09260780781472), (168.06310982824243, 105.13409971966706),
                  (142.8348480301021, 121.25497402689633), (243.49021885852, 35.342282086723486),
                  (233.18810488658843, -173.0188296689552), (-148.7518697402952, 176.45625805690753)]


def get_minimum_bandwidth(index, state):
    power = state[index]

    for i in range(9999):
        bandwidth = (i + 1) * 0.0001
        c = 11.95
        b = 0.136
        alpha_line_of_sight = 2.5
        alpha_none_line_of_sight = 3.5
        beta = 5
        n0 = 10e-17
        noise = bandwidth * n0
        # e = 0.6

        h = uav_height
        d = np.sqrt(user_positions[index][0] **
                    2 + user_positions[index][1] ** 2 + h ** 2)

        theta = (180 / np.pi) * np.arcsin(h / d)
        probability_line_of_sight = 1 / (1 + c * np.exp(-b * (theta - c)))
        probability_none_line_of_sight = 1 - \
                                         (1 / (1 + c * np.exp(-b * (theta - c))))
        loss_line_of_sight = np.power(d, -alpha_line_of_sight)
        snr_line_of_sight = power * loss_line_of_sight / noise
        loss_none_line_of_sight = np.power(d, -alpha_none_line_of_sight)
        snr_none_line_of_sight = power * loss_none_line_of_sight / noise
        eta_line_of_sight = bandwidth * np.log2(1 + snr_line_of_sight)
        eta_none_line_of_sight = bandwidth * np.log2(1 + snr_none_line_of_sight)
        eta = eta_line_of_sight * probability_line_of_sight + \
              eta_none_line_of_sight * probability_none_line_of_sight

        need = threshold * np.power(index + 1, 1 / beta)
        if eta >= need:
            return bandwidth


class PowerEnv(gym.Env, ABC):
    def __init__(self):
        super(PowerEnv, self).__init__()
        self.max = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(user_number,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=total_power / user_number, shape=(user_number,),
                                            dtype=np.float32)
        self.state = np.array([total_power / (user_number)] * user_number).astype(np.float32)

    def reset(self, **kwargs):
        self.state = np.array([total_power / (user_number)] * user_number).astype(np.float32)
        return self.state, {}

    def step(self, action):
        self.state += action
        self.state = np.clip(self.state, 0.001, total_power / user_number)
        min_bandwidth = []
        reward = 0
        for i in range(user_number):
            m = get_minimum_bandwidth(i, self.state)
            min_bandwidth.append(m)
        min_bandwidth.sort()
        used_bandwidth = 0
        for i in range(user_number):
            if min_bandwidth[i] + used_bandwidth <= total_bandwidth:
                reward += 1
                used_bandwidth += min_bandwidth[i]
            else:
                break
        self.max = max(self.max, reward)
        print(self.max)
        return self.state, reward, False, False, {}


# training DDPG
ddpg_env = PowerEnv()
n_actions = ddpg_env.action_space.shape[-1]
power_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
ddpg_agent = DDPG("MlpPolicy", ddpg_env, action_noise=power_action_noise, verbose=1)
# ddpg_agent = TD3("MlpPolicy", ddpg_env, action_noise=power_action_noise, verbose=1)
ddpg_agent.learn(total_timesteps=1000)
