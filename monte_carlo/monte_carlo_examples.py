from collections import defaultdict

import gymnasium

from monte_carlo.first_visit_mc_prediction import mc_prediction
from monte_carlo.off_policy_mc_control import off_policy_mc_control
from monte_carlo.off_policy_mc_prediction import off_policy_mc_prediction
from monte_carlo.on_policy_first_visit_mc_control import on_policy_mc_control
from utils.utils import get_prob_epsilon_greedy
from utils.plot_blackjack import plot_blackjack_values


def black_jack_mc_prediction():
    env = gymnasium.make("Blackjack-v1")
    state_values = mc_prediction(env, 500000, black_jack_policy)
    plot_blackjack_values(state_values)


def black_jack_mc_control():
    epsilon = 0.01
    env = gymnasium.make("Blackjack-v1")
    q_values = on_policy_mc_control(env, 500000, epsilon)

    state_values = defaultdict(lambda: 0.0)
    for state in q_values:
        for action in range(env.action_space.n):
            state_values[state] += \
                get_prob_epsilon_greedy(state, action, epsilon, q_values, env.action_space.n) * q_values[state][action]

    plot_blackjack_values(state_values)


def black_jack_off_policy_mc_prediction():
    epsilon = 0.1
    env = gymnasium.make("Blackjack-v1")
    q_values = off_policy_mc_prediction(env, 2000000, black_jack_policy_probs, epsilon)

    state_values = defaultdict(lambda: 0.0)
    for state in q_values:
        for action in range(env.action_space.n):
            state_values[state] += \
                black_jack_policy_probs(state, action) * q_values[state][action]

    plot_blackjack_values(state_values)


def black_jack_off_policy_mc_control():
    epsilon = 0.01
    env = gymnasium.make("Blackjack-v1")
    q_values = off_policy_mc_control(env, 2000000, epsilon)

    state_values = defaultdict(lambda: 0.0)
    for state in q_values:
        state_values[state] = q_values[state].max()

    plot_blackjack_values(state_values)


def black_jack_policy(state):
    return 0 if state[0] == 20 or state[0] == 21 else 1


def black_jack_policy_probs(state, action):
    if state[0] == 20 or state[0] == 21:
        return 1.0 if action == 0 else 0.0
    else:
        return 0.0 if action == 0 else 1.0
