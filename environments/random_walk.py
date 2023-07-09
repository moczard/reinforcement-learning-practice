import numpy as np

from utils.environment import Environment


class RandomWalk(Environment):
    def __init__(self, n_states):
        super(RandomWalk, self).__init__(n_states, 2)
        self.start_state = int(n_states / 2)
        self.terminal_states = [0, n_states - 1]
        self.current_state = self.start_state
        self.true_values = [(i + 1) / (self.n_states - 1) for i in range(self.n_states - 2)]

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        if self.current_state in self.terminal_states:
            return self.current_state, 0, True

        self.current_state = self.current_state - 1 if action == 0 else self.current_state + 1

        if self.current_state in self.terminal_states:
            if self.current_state == 0:
                return self.current_state, 0, True
            else:
                return self.current_state, 1, True

        return self.current_state, 0, False

    def get_rmse(self, state_values):
        return np.sqrt(np.mean((state_values[1:-1] - self.true_values) ** 2))

    def policy(self, state):
        return np.random.randint(2)
