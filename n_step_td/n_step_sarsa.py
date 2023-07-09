from collections import defaultdict

import numpy as np

from utils.utils import get_epsilon_greedy_action


def n_step_sarsa(env, iterations, n, epsilon=0.1, alpha=0.4, gamma=1):
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    sum_rewards = []

    for _ in range(iterations):
        states = []
        rewards = [0]
        sum_reward = 0
        actions = []
        state, _ = env.reset()
        states.append(state)
        action = get_epsilon_greedy_action(state, epsilon, env.action_space.n, q_values)
        actions.append(action)

        T = float('inf')
        t = 0
        while True:
            if t < T:
                next_state, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                sum_reward += reward
                states.append(next_state)
                if done:
                    T = t + 1
                else:
                    action = get_epsilon_greedy_action(next_state, epsilon, env.action_space.n, q_values)
                    actions.append(action)

            tau = t - n + 1
            if tau >= 0:
                g = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    g = g + pow(gamma, i - tau - 1) * rewards[i]
                if tau + n < T:
                    g = g + pow(gamma, n) * q_values[states[tau + n]][actions[tau + n]]
                q_values[states[tau]][actions[tau]] = q_values[states[tau]][actions[tau]] + alpha * (g - q_values[states[tau]][actions[tau]])

            if tau == T - 1:
                break

            t += 1

        sum_rewards.append(sum_reward)

    return q_values, sum_rewards

