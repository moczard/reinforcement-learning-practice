from collections import deque

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical

from policy_gradient_methods.policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Reinforce:
    def __init__(self, env):
        self.policy_network = Policy(env.observation_space.shape[0], env.action_space.n).to(device)
        self.env = env

    def train(self, iterations, alpha, gamma):
        optimizer = optim.Adam(self.policy_network.parameters(), alpha)
        sum_rewards = np.zeros(50)
        epsilon = np.finfo(np.float32).eps.item()

        for i in range(iterations):
            state, _ = self.env.reset()
            terminated = False
            log_probs = []
            rewards = []
            sum_reward = 0

            while not terminated:
                action, log_prob = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                sum_reward += reward
                state = next_state

            sum_rewards[i % 50] = sum_reward
            g = 0
            returns = deque()
            for reward in reversed(rewards):
                g = gamma * g + reward
                returns.appendleft(g)

            returns = np.array(returns)
            returns = (returns - returns.mean()) / (returns.std() + epsilon)

            loss = []
            for r, l_prob in zip(returns, log_probs):
                loss.append(-l_prob * r)

            episode_loss = torch.cat(loss).sum()
            optimizer.zero_grad()
            episode_loss.backward()
            optimizer.step()

            if i >= 49:
                print(sum_rewards.mean())

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy_network(state)
        m = Categorical(probs)

        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob
