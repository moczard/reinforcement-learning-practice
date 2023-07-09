import numpy as np


class TrueOnlineTDLambda:
    def __init__(self, env, get_feature_vector, w_dimension):
        self.get_feature_vector = get_feature_vector
        self.env = env
        self.d = w_dimension

    def train(self, iterations, policy, alpha, trace_decay_rate, gamma=1):
        w = np.zeros(self.d)

        for i in range(iterations):
            state = self.env.reset()
            x = self.get_feature_vector(state)
            terminated = False
            z = np.zeros(self.d)
            V_old = 0

            while not terminated:
                next_state, reward, terminated = self.env.step(policy(state))
                x_new = self.get_feature_vector(next_state)
                V = w.T @ x
                V_new = w.T @ x_new
                delta = reward + gamma * V_new - V
                z = gamma * trace_decay_rate * z + (1 - alpha * gamma * trace_decay_rate * (z.T @ x)) * x

                w = w + alpha * (delta + V - V_old) * z - alpha * (V - V_old) * x
                V_old = V_new
                x = x_new
                state = next_state

        return w
