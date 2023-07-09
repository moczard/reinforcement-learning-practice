from collections import defaultdict

from utils.utils import generate_episode


class GradientMCPrediction:
    def __init__(self, env, v_hat, delta_v_hat, init_weights):
        self.v_hat = v_hat
        self.delta_v_hat = delta_v_hat
        self.env = env
        self.w = init_weights

    def train(self, iterations, policy, alpha, gamma=1):
        for i in range(iterations):
            episode = generate_episode(self.env, policy)
            returns = defaultdict()

            states, actions, rewards = zip(*episode)
            states = list(states)
            g = 0
            for t in reversed(range(len(episode))):
                g = gamma * g + rewards[t]
                returns[t] = g

            for t in range(len(episode)):
                self.w = self.w + alpha * ((returns[t] - self.v_hat(states[t], self.w)) * self.delta_v_hat(states[t], self.w))

        return self.w
