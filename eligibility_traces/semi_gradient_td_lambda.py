import numpy as np


class SemiGradientTDLambda:
    def __init__(self, env, v_hat, delta_v_hat, w_dimension):
        self.v_hat = v_hat
        self.delta_v_hat = delta_v_hat
        self.env = env
        self.d = w_dimension

    def train(self, iterations, policy, alpha, trace_decay_rate, gamma=1):
        w = np.zeros(self.d)

        for i in range(iterations):
            state = self.env.reset()
            terminated = False
            z = np.zeros(self.d)

            while not terminated:
                next_state, reward, terminated = self.env.step(policy(state))

                z = gamma * trace_decay_rate * z + self.delta_v_hat(state, w)
                delta = reward + gamma * self.v_hat(next_state, w) - self.v_hat(state, w)
                w = w + alpha * delta * z

                state = next_state

        return w
