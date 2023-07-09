from environments.grid_world import GridWorld


class CliffWalking(GridWorld):
    def __init__(self):
        super(CliffWalking, self).__init__(12, 4, [47], 36)

    def step(self, action):
        super().step(action)

        if self.current_state in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
            self.current_state = self.start_state
            return self.current_state, -100, False
        elif self.current_state not in self.terminal_states:
            return self.current_state, -1, False
        else:
            return self.current_state, 0, True
