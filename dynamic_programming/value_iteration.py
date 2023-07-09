import numpy as np


class ValueIteration:
    def __init__(self, env, start_policy, gamma=1):
        self.gamma = gamma
        self.env = env
        self.state_values = np.zeros(env.n_states)
        self.policy = start_policy

    def value_iteration(self):
        theta = 0.00001
        delta = theta + 1
        while delta > theta:
            delta = 0
            for state in range(self.env.n_states):
                if state not in self.env.terminal_states:
                    v_old = self.state_values[state]
                    max_action_value, max_action = self.get_max_action_value(state)
                    self.state_values[state] = max_action_value
                    self.policy[state] = max_action
                    delta = max(delta, abs(v_old - self.state_values[state]))

        return self.policy

    def get_max_action_value(self, state):
        max_action_value = float('-inf')
        max_action = 0
        actions = self.env.get_actions(state)
        for action in actions:
            transitions = self.env.get_transitions(state, action)
            action_value = 0
            for next_state, reward, prob in transitions:
                action_value += prob * (reward + self.gamma * self.state_values[next_state])
            if action_value >= max_action_value:
                max_action_value = action_value
                max_action = action

        return max_action_value, max_action
