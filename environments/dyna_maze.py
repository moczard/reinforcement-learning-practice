from environments.grid_world import GridWorld


class DynaMaze(GridWorld):
    def __init__(self):
        super(DynaMaze, self).__init__(9, 6, [8], 18)
        self.wall = [7, 16, 25, 11, 20, 29, 41]

    def step(self, action):
        transitions = super().get_transitions(self.current_state, action)
        next_state = transitions[0][0]

        if next_state in self.wall:
            return self.current_state, 0, False
        elif next_state not in self.terminal_states:
            self.current_state = next_state
            return self.current_state, 0, False
        else:
            self.current_state = next_state
            return self.current_state, 1, True
