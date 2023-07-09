import numpy as np


class TrueOnlineSarsaLambda:
    def __init__(self, env, tiling):
        self.env = env
        self.tiling = tiling

    def train(self, iterations, epsilon, alpha, decay_rate, gamma=0.95):
        w = np.zeros(self.tiling.get_size())
        rewards = []
        avg_rewards = []

        for i in range(iterations):
            state, _ = self.env.reset()
            action = self.get_epsilon_greedy_action(epsilon, state, w)
            x = self.tiling.get_tiles(state, [action])
            z = np.zeros(self.tiling.get_size())
            q_old = 0
            sum_reward = 0
            avg_reward = 0

            while True:
                next_state, reward, terminated, _, _ = self.env.step(action)
                sum_reward += reward
                next_action = self.get_epsilon_greedy_action(epsilon, next_state, w)
                x_next = self.tiling.get_tiles(next_state, [next_action])
                q = w.take(x).sum()
                q_next = w.take(x_next).sum()
                delta = reward + gamma * q_next - q

                z_delta = 1 - alpha * gamma * decay_rate * z.take(x).sum()
                z = gamma * decay_rate * z
                for tile_index in x:
                    z[tile_index] += z_delta

                w = w + (alpha * (delta + q - q_old)) * z
                w_delta = alpha * (q - q_old)
                for tile_index in x:
                    w[tile_index] -= w_delta

                q_old = q_next
                x = x_next
                action = next_action

                if terminated:
                    break

            rewards.append(sum_reward)
            if i >= 100:
                avg_reward = sum(rewards[i - 100:i]) / 100
            avg_rewards.append(avg_reward)

        return rewards, avg_rewards

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
