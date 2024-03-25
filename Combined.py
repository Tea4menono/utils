# !/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '
__author__ = 'Xuli Cai'

import math
import random
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.noise import NormalActionNoise

# global configs
uav_height = 200
radius = 300
threshold = 0.2
user_number = 100
total_power = 1
total_bandwidth = 1

band = []
power = []

# set up logger
user_positions = [(radius * np.sqrt(np.random.uniform(0, 1)) * np.cos(np.random.uniform(0, 2 * np.pi)),
                   radius * np.sqrt(np.random.uniform(0, 1)) * np.sin(np.random.uniform(0, 2 * np.pi)))
                  for _ in range(user_number)]

thresholds = [threshold * np.power(random.randint(0, 99) + 1, 1 / 5)
              for _ in range(user_number)]


class BandwidthAllocationEnv(gym.Env):
    def __init__(self):
        super(BandwidthAllocationEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0.0001, 0]),
                                            high=np.array(
                                                [total_power / user_number, total_bandwidth, user_number - 1]),
                                            dtype=float)  # Power and Bandwidth
        self.state = np.array(
            [np.random.uniform(0, total_power / user_number), 0.0001,
             np.random.uniform(0, user_number - 1)],
            dtype=np.float32)

    def set(self, power, index):
        self.state = np.array([power, 0.0001, index],
                              dtype=np.float32)  # Random initial power, no bandwidth allocated
        return self.state

    def reset(self, **kwargs):
        self.state = np.array([np.random.uniform(0.0001, total_power), 0.0001,
                               np.random.randint(0, user_number - 1)],
                              dtype=np.float32)
        return self.state, {}

    def step(self, action):
        delta_bandwidth = 0.0001

        if action == 0:
            self.state[1] -= delta_bandwidth
        if action == 2:
            self.state[1] += delta_bandwidth
        self.state[1] = np.clip(self.state[1], 0.0001, total_bandwidth)
        eta = get_eta(self.state[0], self.state[1], int(self.state[2]))
        eta_required = thresholds[int(self.state[2])]

        ratio = eta / eta_required

        if ratio >= 1:
            reward = -np.log(ratio - 1)
        else:
            reward = ratio

        done = 1 <= ratio

        # if len(band) <= 100000:
        # band.append(ratio)
        return self.state, reward, done, False, {}


def get_eta(power, bandwidth, index):
    c = 11.95
    b = 0.136
    alpha_line_of_sight = 2.5
    alpha_none_line_of_sight = 3.5
    n0 = 10e-17
    noise = bandwidth * n0

    d = np.sqrt(user_positions[index][0] ** 2 + user_positions[index][1] ** 2 + uav_height ** 2)

    theta = (180 / np.pi) * np.arcsin(uav_height / d)
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

    return eta


def get_bandwidth(power, index):
    bandwidth = 0
    for i in range(9999):
        bandwidth += 0.0001
        eta = get_eta(power, bandwidth, index)
        eta_required = thresholds[index]
        done = 1 <= (eta / eta_required)
        if done:
            return bandwidth
    return 1


def check(power, bandwidth):
    res = 0
    for i in range(len(power)):
        if bandwidth[i] == 0:
            continue
        eta = get_eta(power[i], bandwidth[i], i)
        eta_required = thresholds[i]
        done = 1 <= (eta / eta_required)
        if done:
            res += 1
    return res


# Create the environment
dqn_env = BandwidthAllocationEnv()
# Instantiate the DQN agent
model = DQN("MlpPolicy", dqn_env, verbose=1)
# Train the agent
model.learn(total_timesteps=100000)
print("Training finished")
# Save the model
model.save("dqn_custom_env")
# Load the model
loaded_model = DQN.load("dqn_custom_env")
print("Model loaded successfully")


# plt.figure(figsize=(30, 10))
# plt.title('DQN minimum bandwidth')
# plt.xlabel('Epochs')
# plt.ylabel('Ratio')
#
# plt.plot(band)
# plt.show()


class PowerEnv(gym.Env):
    def __init__(self):
        super(PowerEnv, self).__init__()
        self.max = -math.inf
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(user_number,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'power': spaces.Box(low=0, high=total_power / user_number, shape=(user_number,), dtype=np.float32),
            'bandwidth': spaces.Box(low=0, high=total_bandwidth, shape=(user_number,), dtype=np.float32)
        })
        # Initialize the state with separate arrays for power and bandwidth
        self.state = {
            'power': np.array([0.0001] * user_number).astype(np.float32),
            'bandwidth': np.array([total_bandwidth / user_number] * user_number).astype(np.float32)
        }

    def reset(self, **kwargs):
        self.state = {
            'power': np.array([0.0001] * user_number).astype(np.float32),
            'bandwidth': np.array([total_bandwidth / user_number] * user_number).astype(np.float32)
        }
        return self.state, {}

    def step(self, action):
        self.state['power'] += action
        self.state['power'] = np.clip(self.state['power'], 0.0001, 2 * total_power / user_number)

        min_bandwidth = []

        # print("before optimization reward:", check(self.state['power'], self.state['bandwidth']))

        for i in range(user_number):
            obs = dqn_env.set(self.state['power'][i], i)
            for _ in range(1000):
                actions, _states = loaded_model.predict(obs, deterministic=True)
                obs, rewards, dones, truncated, info = dqn_env.step(actions)
                if _ > 500:
                    min_bandwidth.append(get_bandwidth(self.state['power'][i], i))
                    break
                if dones:
                    min_bandwidth.append(obs[1])
                    break

        self.state['bandwidth'] = np.array([0] * user_number).astype(np.float32)
        used_bandwidth = 0
        reward = 0

        while True:
            # move
            min_value = np.min(min_bandwidth)
            min_index = np.argmin(min_bandwidth)

            if used_bandwidth + min_value <= total_bandwidth:
                # add to total
                used_bandwidth += min_value
                self.state['bandwidth'][min_index] = min_value
                reward += 1
                # clear
                min_bandwidth[min_index] = math.inf
            else:
                break

        self.max = max(self.max, reward)
        overload = sum(self.state['power']) >= total_power
        if overload:
            reward = reward - (sum(self.state['power']) - total_power) * 10
        print("max:", self.max, "reward:", reward, "total", sum(self.state['power']))

        if len(power) < 100:
            power.append(reward)
        return self.state, reward, len(power) >= 100, False, {}


# training DDPG
ddpg_env = PowerEnv()
n_actions = ddpg_env.action_space.shape[-1]
power_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
ddpg_agent = DDPG("MultiInputPolicy", ddpg_env, action_noise=power_action_noise, verbose=1)
# ddpg_agent = TD3("MlpPolicy", ddpg_env, action_noise=power_action_noise, verbose=1)
ddpg_agent.learn(total_timesteps=1)

plt.figure(figsize=(30, 10))
plt.title('DDPG served user')
plt.xlabel('Epochs')
plt.ylabel('User number')

plt.plot(power)
plt.show()
