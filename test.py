import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise


class SimpleContinuousEnv(gym.Env):
    def __init__(self, state_range=(1, 1)):
        super(SimpleContinuousEnv, self).__init__()
        self.state_range = state_range
        self.observation_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.state = np.array([0, 0])

    def step(self, action):
        self.state += action
        self.state = np.clip(
            self.state,
            # Minimum values for x and y
            [self.state_range[0], self.state_range[0]],
            # Maximum values for x and y
            [self.state_range[1], self.state_range[1]]
        )

        reward = np.linalg.norm(self.state)
        print(f'state = {self.state},action = {action}, reward = {reward}')
        done = False
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([0.0, 0.0])
        return self.state

    def render(self, mode='human'):
        pass


env = SimpleContinuousEnv()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(
    n_actions), sigma=0.1 * np.ones(n_actions))

# Create the DDPG agent
ddpg_agent = DDPG(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,
    buffer_size=100000,
    batch_size=100,
    gamma=0.99,
    tau=0.005,
    action_noise=action_noise,
    verbose=1,
)

# Train the agent
episodes = 100
for e in range(episodes):
    state = env.reset()
    print("==========")
    for time in range(200):
        action = ddpg_agent.predict(state, deterministic=True)[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

# Test the trained agent
state = env.reset()
for time in range(200):
    action = ddpg_agent.predict(state, deterministic=True)[0]
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break

print("Final state:", state)
