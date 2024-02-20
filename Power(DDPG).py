import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import random
from env import UAV


class Config:
    def __init__(self):
        self.train_eps = 100
        self.max_steps = 200
        self.batch_size = 512
        self.memory_capacity = 10000
        self.lr_a = 1e-4  # Learning rate for the actor
        self.lr_c = 1e-4  # Learning rate for the critic
        self.gamma = 0.99
        self.sigma = 0.005
        self.tau = 0.01
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.seed = random.randint(0, 100)
        self.n_states = None
        self.n_actions = None
        self.action_high_bound = None
        self.action_low_bound = None
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

    def show(self):
        print('-' * 30 + 'Parameters List' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)


class ReplayBuffer:
    def __init__(self, cfg):
        self.buffer = np.empty(cfg.memory_capacity, dtype=object)
        self.size = 0
        self.pointer = 0
        self.capacity = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    def push(self, transitions):
        self.buffer[self.pointer] = transitions
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.size = 0
        self.pointer = 0

    def sample(self):
        batch_size = min(self.batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.actor_hidden_dim)
        self.bn1 = nn.BatchNorm1d(cfg.actor_hidden_dim)
        self.fc2 = nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim)
        self.bn2 = nn.BatchNorm1d(cfg.actor_hidden_dim)
        self.fc3 = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * \
            torch.FloatTensor(cfg.action_high_bound) * 0.5
        return action


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states + cfg.n_actions,
                             cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, 1)
        self.apply(init_weights)  # Apply weight initialization

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DDPG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg)
        self.actor = Actor(cfg).to(cfg.device)

        self.actor_target = Actor(cfg).to(cfg.device)
        self.critic = Critic(cfg).to(cfg.device)
        self.critic_target = Critic(cfg).to(cfg.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_a)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_c)
        self.critic_target.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.cfg.device).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().numpy()
        action += self.cfg.sigma * np.random.randn(self.cfg.n_actions)
        noisy_action = np.clip(
            action, self.cfg.action_low_bound, self.cfg.action_high_bound)

        return noisy_action

    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return 0, 0
        states, actions, rewards, next_states, dones = self.memory.sample()

        actions, rewards, dones = actions.view(-1,
                                               41), rewards.view(-1, 1), dones.view(-1, 1)
        next_q_value = self.critic_target(
            next_states, self.actor_target(next_states))
        target_q_value = rewards + (1 - dones) * self.cfg.gamma * next_q_value

        critic_loss = torch.mean(F.mse_loss(
            self.critic(states, actions), target_q_value))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # Apply gradient clipping to Critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # Apply gradient clipping to Actor
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optim.step()

        self.update_params()

        return actor_loss.item(), critic_loss.item()

    def update_params(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)


def env_agent_config(cfg):
    env = UAV()
    print(f'observation = {env.observation_space}')
    print(f'action = {env.action_space}')
    cfg.n_states = env.observation_space.shape[0]
    cfg.n_actions = env.action_space.shape[0]
    cfg.action_high_bound = env.action_space.high
    cfg.action_low_bound = env.action_space.low

    agent = DDPG(cfg)
    return env, agent


def train(env, agent, cfg):
    print('Start Training!')
    cfg.show()
    rewards, steps, critic_losses, actor_losses = [], [], [], []
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        critic_loss, actor_loss = 0.0, 0.0
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated
            # if ep_step == 1:
            # print(reward, "reward")
            # print(state[1:21], "power state")
            # print(state[21:], "bw state")
            # print(action[1:21], "power action")
            # print(action[21:], "bw action")
            # print(next_state[1:21], "power next_state")
            # print(next_state[21:], "bw next_state")
            # print("====================================")

            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            c_loss, a_loss = agent.update()
            critic_loss += c_loss
            actor_loss += a_loss
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward/ep_step)
        steps.append(ep_step)
        critic_losses.append(critic_loss/ep_step)
        actor_losses.append(actor_loss/ep_step)
        print(f'Episode:{i + 1}/{cfg.train_eps}  Reward:{ep_reward/ep_step:.4f}  Steps:{ep_step:.0f}'
              f'  Critic Loss:{critic_loss/ep_step:.4f}  Actor Loss:{actor_loss/ep_step:.4f}')
    print('Finish Training!')
    env.close()
    return rewards, critic_losses, actor_losses, steps


if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    rewards, critic_losses, actor_losses, steps = train(env, agent, cfg)
    plt.figure(figsize=(12, 5))
    plt.plot(rewards)
    plt.title('Traning Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
