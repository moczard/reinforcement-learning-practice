from utils.environment import Environment


class GamblerEnv(Environment):
    def __init__(self, p_head):
        super(GamblerEnv, self).__init__(101, 100)
        self.terminal_states = [0, 100]
        self.p_head = p_head

    def get_transitions(self, state, action):
        losing_state = 0 if state - action <= 0 else state - action
        winning_state = 100 if state + action >= 100 else state + action
        losing_reward = 0
        winning_reward = 1 if winning_state == 100 else 0
        return [(losing_state, losing_reward, 1 - self.p_head), (winning_state, winning_reward, self.p_head)]

    def get_actions(self, state):
        return range(min(state, 100 - state) + 1)
