import numpy as np

from environments.grid_world import GridWorld


def policy_evaluation(env, policy, gamma=1):
    theta = 0.0001
    delta = theta + 1
    state_values = np.zeros(env.n_states)
    while delta > theta:
        delta = 0
        for state in range(env.n_states):
            v = state_values[state]
            value = 0
            # sum actions weighted with action probabilities
            for action in range(env.n_actions):
                transitions = env.get_transitions(state, action)
                # sum next_states and rewards weighted with transition probabilities
                for next_state, reward, transition_prob in transitions:
                    value += policy[state][action] * transition_prob * (reward + gamma * state_values[next_state])
            state_values[state] = value
            delta = max(delta, abs(v - state_values[state]))

    return state_values



