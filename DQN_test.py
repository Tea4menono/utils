import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN

# global configs
height = 200
radius = 300
threshold = 0.5
user_number = 10
total_power = 1
total_bandwidth = 1

# set up logger
user_positions = [(radius * np.sqrt(np.random.uniform(0, 1)) * np.cos(np.random.uniform(0, 2 * np.pi)),
                   radius * np.sqrt(np.random.uniform(0, 1)) * np.sin(np.random.uniform(0, 2 * np.pi)))
                  for _ in range(user_number)]

uav_bandwidth = [0] * user_number
uav_power = [0] * user_number


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
    def __init__(self, index):
        super(BandwidthEnv, self).__init__()
        self.index = index
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([total_bandwidth, total_power]),
                                            shape=(2,),
                                            dtype=np.float32)
        self.state = np.array([0.0005, 0.0005], dtype=np.float32)

    def step(self, action):
        delta_bandwidth = 0.0001
        delta_power = 0.0001
        action_map = {
            0: [delta_bandwidth, delta_power],
            1: [delta_bandwidth, 0],
            2: [delta_bandwidth, -delta_power],
            3: [0, delta_power],
            4: [0, 0],
            5: [0, -delta_power],
            6: [-delta_bandwidth, delta_power],
            7: [-delta_bandwidth, 0],
            8: [-delta_bandwidth, -delta_power]
        }

        self.state += np.array(action_map[action], dtype=np.float32)
        self.state = np.clip(self.state, [0.0001, 0.0001], [total_bandwidth, total_power])

        ratio = calculate_reward(self.state[0], self.state[1], self.index)
        done = 1 <= ratio <= 1.05
        penalty = self.state[0] + self.state[1]
        reward = ratio - penalty

        uav_bandwidth[self.index] = self.state[0]
        uav_power[self.index] = self.state[1]

        return self.state.copy(), reward, done, False, {}

    def reset(self, **kwargs):
        self.state = np.array([0.0005, 0.0005], dtype=np.float32)


# training DQN
dqn_env = BandwidthEnv(0)
dqn_model = DQN("MlpPolicy", dqn_env, verbose=1)
dqn_model.learn(total_timesteps=100000)


def test_model(env, model, num_episodes=10):
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = model.predict(observation, deterministic=True)[0]
            observation, reward, done, truncated, info = env.step(action.min())
            print(action, observation, reward, done)
            if done:
                print(f"Episode {episode} finished")
                print(observation, reward)
                break


# Test the trained DQN model
test_env = BandwidthEnv(0)
test_model(test_env, dqn_model, 1)
