import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# global configs
height = 200
radius = 300
threshold = 0.1
user_number = 20
total_power = 1
total_bandwidth = 1

# set up logger
user_positions = [(radius * np.sqrt(np.random.uniform(0, 1)) * np.cos(np.random.uniform(0, 2 * np.pi)),
                   radius * np.sqrt(np.random.uniform(0, 1)) * np.sin(np.random.uniform(0, 2 * np.pi)))
                  for _ in range(user_number)]


def calculate_reward(bandwidth, power, index):
    c = 11.95
    b = 0.136
    alpha_line_of_sight = 2.5
    alpha_none_line_of_sight = 3.5
    beta = 5
    n0 = 10e-17
    noise = bandwidth * n0
    e = 0.6

    h = height
    d = np.sqrt(user_positions[index][0] **
                2 + user_positions[index][1] ** 2 + h ** 2)

    # degrees_of_freedom = 2

    # fading
    # snr_line_of_sight = np.random.chisquare(
    #     degrees_of_freedom) * self.uav_power[i] * np.power(d, -alpha) / noise
    # snr_none_line_of_sight = np.random.exponential(0.1) * e * \
    #     self.uav_power[i] * np.power(d, -alpha) / noise

    theta = (180 / np.pi) * np.arcsin(h / d)
    probability_line_of_sight = 1 / (1 + c * np.exp(-b * (theta - c)))
    probability_none_line_of_sight = 1 - \
                                     (1 / (1 + c * np.exp(-b * (theta - c))))
    loss_line_of_sight = np.power(d, -alpha_line_of_sight)
    snr_line_of_sight = power * loss_line_of_sight / noise
    loss_none_line_of_sight = np.power(d, -alpha_none_line_of_sight)
    snr_none_line_of_sight = e * power * loss_none_line_of_sight / noise
    eta_line_of_sight = bandwidth * np.log2(1 + snr_line_of_sight)
    eta_none_line_of_sight = bandwidth * np.log2(1 + snr_none_line_of_sight)
    eta = eta_line_of_sight * probability_line_of_sight + \
          eta_none_line_of_sight * probability_none_line_of_sight
    need = threshold * np.power(index + 1, 1 / beta)
    return eta / need


class BandwidthEnv(gym.Env):
    def __init__(self):
        super(BandwidthEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([0.1, 0.1]),
                                       shape=(2,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
                                            high=np.array([total_bandwidth, total_power, user_number]),
                                            shape=(3,),
                                            dtype=np.float32)
        self.state = np.array([0, 0, 0], dtype=np.float32)
        self.reward = 0

    def step(self, action):
        truncated = False
        done = False
        penalty = 0

        if self.state[2] >= user_number:
            done = True
        else:
            ratio = calculate_reward(action[0], action[1], int(self.state[2]))
            if ratio >= 1:
                self.reward += 1

            self.state += np.array([action[0], action[1], 1], dtype=np.float32)

        # go beyond the bound
        if self.state[0] > total_bandwidth:
            penalty = self.state[0] - total_bandwidth
        if self.state[1] > total_power:
            penalty += self.state[1] - total_power

        print(self.reward - penalty * 10)
        return self.state.copy(), self.reward - penalty * 10, done, truncated, {}

    def reset(self, **kwargs):
        self.state = np.array([0, 0, 0], dtype=np.float32)
        self.reward = 0
        return self.state, {}


ddpg_env = BandwidthEnv()
n_actions = ddpg_env.action_space.shape[-1]
power_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
ddpg_agent = DDPG("MlpPolicy", ddpg_env, action_noise=power_action_noise, verbose=1)
ddpg_agent.learn(total_timesteps=10000)


def test_model(env, model, num_episodes=1):
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = env.step(action)
            if done:
                print(f"Episode {episode} finished")
                print(observation, reward)
                break


# Test the trained DQN model
test_env = BandwidthEnv()
test_model(test_env, ddpg_agent, 1)
