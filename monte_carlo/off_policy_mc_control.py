from collections import defaultdict

import numpy as np

from utils.utils import generate_episode_epsilon_greedy, get_prob_epsilon_greedy


def off_policy_mc_control(env, iterations, epsilon=0.01, gamma=1):
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    c = defaultdict(lambda: np.zeros(env.action_space.n))
    for i in range(iterations):
        episode = generate_episode_epsilon_greedy(env, epsilon, q_values)
        states, actions, rewards = zip(*episode)
        states = list(states)
        g = 0
        w = 1
        for t in reversed(range(len(episode))):
            g = gamma * g + rewards[t]
            c[states[t]][actions[t]] += w
            q_values[states[t]][actions[t]] += (w / c[states[t]][actions[t]]) * (g - q_values[states[t]][actions[t]])
            max_action = q_values[states[t]].argmax()
            if actions[t] != max_action:
                break
            w = w * (1 / get_prob_epsilon_greedy(states[t], actions[t], epsilon, q_values, env.action_space.n))

    return q_values
