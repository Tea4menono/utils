import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.noise import NormalActionNoise


class CombinedAllocationEnv(gym.Env):
    def __init__(self):
        super(CombinedAllocationEnv, self).__init__()
        self.observation_space = spaces.Box(low=np.array(
            [0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1]), dtype=np.float32)
        self.bandwidth_action_space = spaces.Discrete(3)
        self.power_action_space = spaces.Box(
            low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.state = None

    def step(self, action):
        bandwidth_action, power_action = action
        bandwidth_state_update = bandwidth_action
        power_state_update = power_action
        new_state = self.state + \
            np.array([bandwidth_state_update, 0, 0,
                     power_state_update[0], 0, 0])
        reward = np.random.random()
        done = False
        info = {}
        return new_state, reward, done, info

    def reset(self):
        self.state = np.random.random(6)
        return self.state

    def render(self, mode='human'):
        pass


env = CombinedAllocationEnv()

# DQN for discrete bandwidth allocation
dqn_agent = DQN(
    policy="MlpPolicy",
    env=env,  # Note: You would need to adapt the environment to provide only the bandwidth-related state and actions
    learning_rate=0.001,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
)

# DDPG for continuous power allocation
n_actions = env.power_action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(
    n_actions), sigma=0.1 * np.ones(n_actions))

ddpg_agent = DDPG(
    policy="MlpPolicy",
    env=env,  # Note: You would need to adapt the environment to provide only the power-related state and actions
    learning_rate=0.001,
    buffer_size=100000,
    batch_size=100,
    gamma=0.99,
    tau=0.005,
    action_noise=action_noise,
    verbose=1,
)

episodes = 100
for e in range(episodes):
    state = env.reset()
    for time in range(500):
        # DQN action (bandwidth)
        bandwidth_action = dqn_agent.predict(state, deterministic=True)[0]
        # DDPG action (power)
        power_action = ddpg_agent.predict(state, deterministic=True)[0]
        # Combine actions
        combined_action = (bandwidth_action, power_action)
        next_state, reward, done, _ = env.step(combined_action)
        state = next_state
        if done:
            break
