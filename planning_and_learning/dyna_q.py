import random
from collections import defaultdict

import numpy as np

from utils.utils import get_epsilon_greedy_action_random_ties


def dyna_q(env, iterations, n, epsilon=0.1, alpha=0.1, gamma=0.95):
    q_values = defaultdict(lambda: np.zeros(env.n_actions))
    model = defaultdict(lambda: defaultdict())
    step_counter = []

    for _ in range(iterations):
        state = env.reset()
        terminated = False
        step = 0

        while not terminated:
            action = get_epsilon_greedy_action_random_ties(epsilon, env.n_actions, q_values[state])
            next_state, reward, terminated = env.step(action)
            step += 1

            q_values[state][action] = q_values[state][action] + alpha * (reward + gamma * np.amax(q_values[next_state]) - q_values[state][action])

            model[state][action] = (next_state, reward)
            state = next_state

            for _ in range(n):
                planning_state = random.choice(list(model.keys()))
                planning_action = random.choice(list(model[planning_state].keys()))
                planning_next_state, planning_reward = model[planning_state][planning_action]

                q_values[planning_state][planning_action] = q_values[planning_state][planning_action] + alpha * (planning_reward + gamma * np.amax(q_values[planning_next_state]) - q_values[planning_state][planning_action])

        step_counter.append(step)

    return q_values, step_counter
