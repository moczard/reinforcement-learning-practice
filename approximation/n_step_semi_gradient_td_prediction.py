class NStepSemiGradientTDPrediction:
    def __init__(self, env, v_hat, delta_v_hat, init_weights):
        self.v_hat = v_hat
        self.delta_v_hat = delta_v_hat
        self.env = env
        self.w = init_weights

    def train(self, iterations, policy, n, alpha, gamma=1):
        for _ in range(iterations):
            states = []
            rewards = [0]
            state, _ = self.env.reset()
            states.append(state)

            T = float('inf')
            t = 0
            while True:
                if t < T:
                    action = policy(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                    rewards.append(reward)
                    states.append(next_state)
                    if done:
                        T = t + 1

                tau = t - n + 1
                if tau >= 0:
                    g = 0.0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        g = g + pow(gamma, i - tau - 1) * rewards[i]
                    if tau + n < T:
                        g = g + pow(gamma, n) * self.v_hat(states[tau + n], self.w)
                    self.w = self.w + alpha * (g - self.v_hat(states[tau], self.w)) * self.delta_v_hat(states[tau], self.w)

                if tau == T - 1:
                    break

                t += 1

        return self.w