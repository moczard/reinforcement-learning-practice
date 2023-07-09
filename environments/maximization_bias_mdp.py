import numpy as np

from utils.environment import Environment


class MaximizationBiasMDP(Environment):
    def __init__(self):
        super(MaximizationBiasMDP, self).__init__(4, 2)
        self.start_state = 2
        self.terminal_states = [0, 3]
        self.current_state = self.start_state

    def reset(self):
        self.current_state = self.start_state
        return self.start_state

    def step(self, action):
        if self.current_state in self.terminal_states:
            return self.current_state, 0, True

        if self.current_state == 2 and action == 1:
            self.current_state = 3
            return self.current_state, 0, True

        if self.current_state == 2 and action == 0:
            self.current_state = 1
            return self.current_state, 0, False

        if self.current_state == 1:
            self.current_state = 0
            return self.current_state, np.random.normal(-0.1, 1.0), True

    def get_possible_actions(self, state):
        if state == 2:
            return [0, 1]
        if state == 1:
            return [i for i in range(10)]

        return []
