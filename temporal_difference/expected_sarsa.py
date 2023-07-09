from collections import defaultdict

import numpy as np

from utils.utils import get_epsilon_greedy_action, get_probs_epsilon_greedy


def expected_sarsa(env, iterations, epsilon=0.1, alpha=0.1, gamma=1.0):
    q_values = defaultdict(lambda: np.zeros(env.n_actions))
    step_counter = []
    sum_rewards = []

    for i in range(iterations):
        state = env.reset()
        terminated = False
        step = 0
        sum_reward = 0

        while not terminated:
            action = get_epsilon_greedy_action(state, epsilon, env.n_actions, q_values)
            next_state, reward, terminated = env.step(action)
            step += 1
            sum_reward += reward

            expected_next_state_value = np.dot(q_values[next_state], get_probs_epsilon_greedy(next_state, epsilon, q_values, env.n_actions))
            q_values[state][action] = q_values[state][action] + alpha * (reward + gamma * expected_next_state_value - q_values[state][action])
            state = next_state

        step_counter.append(step)
        sum_rewards.append(sum_reward)

    return q_values, step_counter, sum_rewards
