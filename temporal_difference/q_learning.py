from collections import defaultdict

import numpy as np

from utils.utils import get_epsilon_greedy_action, get_epsilon_greedy_action_random_ties


def q_learning(env, iterations, epsilon=0.1, alpha=0.5, gamma=1.0):
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

            q_values[state][action] = q_values[state][action] + alpha * (reward + gamma * np.amax(q_values[next_state]) - q_values[state][action])
            state = next_state

        step_counter.append(step)
        sum_rewards.append(sum_reward)

    return q_values, step_counter, sum_rewards


def q_learning_max_bias(env, iterations, epsilon=0.1, alpha=0.5, gamma=1.0):
    q_values = defaultdict(lambda: np.zeros(env.n_actions))
    q_values[1] = np.zeros(10)
    left_action_percentage = []
    left_action_count = 0

    for i in range(iterations):
        state = env.reset()
        terminated = False

        while not terminated:
            actions = env.get_possible_actions(state)
            action = get_epsilon_greedy_action_random_ties(epsilon, actions, q_values[state])
            if state == 2:
                if action == 0:
                    left_action_count += 1
                left_action_percentage.append(left_action_count / (i + 1))

            next_state, reward, terminated = env.step(action)

            q_values[state][action] = q_values[state][action] + alpha * (reward + gamma * np.amax(q_values[next_state]) - q_values[state][action])
            state = next_state

    return left_action_percentage


def double_q_learning_max_bias(env, iterations, epsilon=0.1, alpha=0.5, gamma=1.0):
    q_values_1 = defaultdict(lambda: np.zeros(env.n_actions))
    q_values_1[1] = np.zeros(10)
    q_values_2 = defaultdict(lambda: np.zeros(env.n_actions))
    q_values_2[1] = np.zeros(10)
    left_action_percentage = []
    left_action_count = 0

    for i in range(iterations):
        state = env.reset()
        terminated = False

        while not terminated:
            actions = env.get_possible_actions(state)
            action = get_epsilon_greedy_action_random_ties(epsilon, actions, q_values_1[state] + q_values_2[state])
            if state == 2:
                if action == 0:
                    left_action_count += 1
                left_action_percentage.append(left_action_count / (i + 1))

            next_state, reward, terminated = env.step(action)

            if np.random.randint(2) > 0:
                q_values_1[state][action] = q_values_1[state][action] + alpha * (
                            reward + gamma * q_values_2[next_state][q_values_1[next_state].argmax()] -
                            q_values_1[state][action])
            else:
                q_values_2[state][action] = q_values_2[state][action] + alpha * (
                            reward + gamma * q_values_1[next_state][q_values_2[next_state].argmax()] -
                            q_values_2[state][action])

            state = next_state

    return left_action_percentage
