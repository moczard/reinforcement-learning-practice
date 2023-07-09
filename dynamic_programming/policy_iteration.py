import numpy as np


class PolicyIteration:
    def __init__(self, env, start_policy, gamma=1):
        self.gamma = gamma
        self.env = env
        self.policy = start_policy
        self.state_values = np.zeros(env.n_states)
        self.policy_stable = False

    def policy_evaluation(self):
        theta = 0.01
        delta = theta + 1
        while delta > theta:
            delta = 0
            for state in range(self.env.n_states):
                v_old = self.state_values[state]
                value = 0
                transitions = self.env.get_transitions(state, self.policy[state])
                for next_state, reward, transition_prob in transitions:
                    value += transition_prob * (reward + self.gamma * self.state_values[next_state])
                self.state_values[state] = value
                delta = max(delta, abs(v_old - self.state_values[state]))

    def policy_improvement(self):
        self.policy_stable = True
        for state in range(self.env.n_states):
            old_action = self.policy[state]
            max_action_value = float('-inf')
            max_action = 0
            possible_actions = self.env.get_actions(state)
            for action in possible_actions:
                value = 0
                transitions = self.env.get_transitions(state, action)
                if len(transitions) == 0:
                    continue

                for next_state, reward, transition_prob in transitions:
                    value += transition_prob * (reward + self.gamma * self.state_values[next_state])

                if value > max_action_value:
                    max_action_value = value
                    max_action = action
            self.policy[state] = max_action
            if old_action != max_action:
                self.policy_stable = False

    def policy_iteration(self):
        while not self.policy_stable:
            self.policy_evaluation()
            self.policy_improvement()

        return self.policy
