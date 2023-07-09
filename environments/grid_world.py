from utils.environment import Environment


class GridWorld(Environment):
    def __init__(self, width, height, terminal_states, start_state):
        super(GridWorld, self).__init__(width * height, 4)
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.current_state = start_state
        self.width = width
        self.height = height

    def reset(self):
        self.current_state = self.start_state
        return self.start_state

    def get_transitions(self, state, action):
        if state in self.terminal_states:
            return [(state, 0, 1.0)]

        # UP
        if action == 0:
            next_state = state - self.width if state - self.width >= 0 else state
            return [(next_state, -1, 1.0)]

        # DOWN
        if action == 1:
            next_state = state + self.width if state + self.width < self.n_states else state
            return [(next_state, -1, 1.0)]

        # RIGHT
        if action == 2:
            next_state = state + 1 if state % self.width != self.width - 1 and state < self.n_states else state
            return [(next_state, -1, 1.0)]

        # LEFT
        if action == 3:
            next_state = state - 1 if state % self.width != 0 else state
            return [(next_state, -1, 1.0)]

    def step(self, action):
        transitions = self.get_transitions(self.current_state, action)

        self.current_state = transitions[0][0]
        reward = transitions[0][1]

        return self.current_state, reward, self.current_state in self.terminal_states

    def get_actions(self, state):
        return [0, 1, 2, 3]
