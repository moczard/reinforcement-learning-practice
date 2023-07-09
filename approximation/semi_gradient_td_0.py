class SemiGradientTDPrediction:
    def __init__(self, env, v_hat, delta_v_hat, init_weights):
        self.v_hat = v_hat
        self.delta_v_hat = delta_v_hat
        self.env = env
        self.w = init_weights

    def train(self, iterations, policy, alpha, gamma=1):
        for i in range(iterations):
            state, _ = self.env.reset()
            terminated = False
            while not terminated:
                next_state, reward, terminated, _, _ = self.env.step(policy(state))

                self.w = self.w + alpha * (reward + gamma * self.v_hat(next_state, self.w) - self.v_hat(state, self.w)) * self.delta_v_hat(state, self.w)
                state = next_state

        return self.w
