import numpy as np

from environments.random_walk import RandomWalk


class ThousandStateRandomWalk(RandomWalk):
    def __init__(self):
        super(ThousandStateRandomWalk, self).__init__(1002)
        self.step_range_left = [*range(-100, 0)]
        self.step_range_right = [*range(1, 101)]

    def reset(self):
        self.current_state = self.start_state
        return self.current_state, None

    def step(self, action):
        if self.current_state in self.terminal_states:
            return self.current_state, 0, True

        step_range = self.step_range_left if action == 0 else self.step_range_right
        step = np.random.choice(step_range)

        new_state = self.current_state + step

        if new_state <= 0:
            self.current_state = 0
        elif new_state >= 1001:
            self.current_state = 1001
        else:
            self.current_state = new_state

        if self.current_state in self.terminal_states:
            if self.current_state == 0:
                return self.current_state, -1, True, None, None
            else:
                return self.current_state, 1, True, None, None

        return self.current_state, 0, False, None, None
