import numpy as np


class EpisodicSemiGradientSarsa:
    def __init__(self, env, tiling):
        self.env = env
        self.tiling = tiling

    def train(self, iterations, epsilon, alpha):
        w = np.zeros(self.tiling.get_size())
        rewards = []

        for i in range(iterations):
            state, _ = self.env.reset()
            action = self.get_epsilon_greedy_action(epsilon, state, w)
            sum_reward = 0

            while True:
                next_state, reward, terminated, _, _ = self.env.step(action)
                sum_reward += reward

                if terminated:
                    delta_w = alpha * (float(reward) - self.q_hat(state, action, w))
                    self.update_weights(delta_w, state, action, w)
                    break

                next_action = self.get_epsilon_greedy_action(epsilon, next_state, w)
                delta_w = alpha * (float(reward) + self.q_hat(next_state, next_action, w) - self.q_hat(state, action, w))
                self.update_weights(delta_w, state, action, w)

                state = next_state
                action = next_action

            rewards.append(sum_reward)

        return rewards

    def update_weights(self, delta_w, state, action, w):
        current_tiles = self.tiling.get_tiles(state, [action])
        for tile in current_tiles:
            w[tile] += delta_w

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

    def q_hat(self, state, action, w):
        features = self.tiling.get_tiles(state, [action])
        return w.take(features).sum()
