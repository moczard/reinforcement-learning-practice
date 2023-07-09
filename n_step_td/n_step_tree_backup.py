from collections import defaultdict

import numpy as np

from utils.utils import get_probs_epsilon_greedy, get_epsilon_greedy_action


def n_step_tree_backup(env, iterations, n, epsilon=0.1, alpha=0.4, gamma=1):
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
                if t + 1 >= T:
                    g = rewards[T]
                else:
                    g = rewards[t + 1] + gamma * np.dot(q_values[states[t + 1]], get_probs_epsilon_greedy(states[t + 1], epsilon, q_values, env.action_space.n))

                for k in reversed(range(tau + 1, min(t, T - 1) + 1)):
                    action_probs = get_probs_epsilon_greedy(states[k], epsilon, q_values, env.action_space.n)
                    current_action_prob = action_probs[actions[k]]
                    action_probs[actions[k]] = 0
                    g = rewards[k] + gamma * np.dot(q_values[states[k]], action_probs) + gamma * current_action_prob * g

                q_values[states[tau]][actions[tau]] = q_values[states[tau]][actions[tau]] + alpha * (g - q_values[states[tau]][actions[tau]])

            if tau == T - 1:
                break

            t += 1

        sum_rewards.append(sum_reward)

    return q_values, sum_rewards
