from collections import defaultdict

import numpy as np

from utils.utils import generate_episode_epsilon_greedy


def on_policy_mc_control(env, iterations, epsilon=0.01, gamma=1):
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    sum_returns = defaultdict(lambda: np.zeros(env.action_space.n))
    n_visit = defaultdict(lambda: np.zeros(env.action_space.n))
    for i in range(iterations):
        episode = generate_episode_epsilon_greedy(env, epsilon, q_values)
        states, actions, rewards = zip(*episode)
        states = list(states)
        g = 0
        for t in reversed(range(len(episode))):
            g = gamma * g + rewards[t]
            if states[t] not in states[0:t]:
                sum_returns[states[t]][actions[t]] += g
                n_visit[states[t]][actions[t]] += 1
                q_values[states[t]][actions[t]] = sum_returns[states[t]][actions[t]] / n_visit[states[t]][actions[t]]

    return q_values
