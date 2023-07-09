from environments.grid_world import GridWorld


class WindyGridWorld(GridWorld):
    def __init__(self):
        super(WindyGridWorld, self).__init__(10, 7, [37], 30)

    def step(self, action):
        super().step(action)
        position = self.current_state % 10
        if position == 3 or position == 4 or position == 5 or position == 8:
            super().step(0)
        if position == 6 or position == 7:
            super().step(0)
            super().step(0)

        if self.current_state in self.terminal_states:
            return self.current_state, 0, True
        else:
            return self.current_state, -1, False
