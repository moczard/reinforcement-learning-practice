import numpy as np


class LeastSquaresTD:
    def __init__(self, env, feature_representation, init_weights):
        self.feature_representation = feature_representation
        self.env = env
        self.w = init_weights
        self.d = len(init_weights)

    def train(self, iterations, policy, epsilon, gamma=1):
        I = np.identity(self.d)
        A_inv = (1 / epsilon) * I
        b_hat = np.zeros(self.d)

        for i in range(iterations):
            state, _ = self.env.reset()
            x = self.feature_representation(state)
            terminated = False
            while not terminated:
                next_state, reward, terminated, _, _ = self.env.step(policy(state))
                x_next = self.feature_representation(next_state)

                v = A_inv.T @ (x - gamma * x_next)
                A_inv = A_inv - (np.outer((A_inv @ x), v.T)) / (1 + v.T @ x)
                b_hat = b_hat + reward * x
                self.w = A_inv @ b_hat
                state = next_state
                x = x_next

        return self.w
