import numpy as np


class DifferentialSemiGradientSarsa:
    def __init__(self, env, q_hat, delta_q_hat, init_weights):
        self.q_hat = q_hat
        self.delta_q_hat = delta_q_hat
        self.env = env
        self.w = init_weights

    def train(self, iterations, alpha, beta, epsilon):
        avg_reward = 0

        for i in range(iterations):
            state, _ = self.env.reset()
            action = self.get_epsilon_greedy_action(epsilon, state, self.w)
            terminated = False
            while not terminated:
                next_state, reward, terminated, _, _ = self.env.step(action)
                next_action = self.get_epsilon_greedy_action(epsilon, next_state, self.w)
                delta = reward - avg_reward + self.q_hat(next_state, next_action, self.w) - self.q_hat(state, action, self.w)
                avg_reward = avg_reward + beta * delta
                self.w = self.w + alpha * delta * self.delta_q_hat(state, action, self.w)
                state = next_state
                action = next_action

        return self.w

    def get_epsilon_greedy_action(self, epsilon, state, w):
        if np.random.random() >= epsilon:
            q = float("-inf")
            max_action = self.env.action_space.start
            for action in range(self.env.action_space.n):
                q_new = self.q_hat(state, action, w)
                if q_new > q:
                    q = q_new
                    max_action = action

            return max_action
        else:
            return self.env.action_space.sample()