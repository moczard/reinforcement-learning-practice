import numpy as np


class EpisodicSemiGradientNStepSarsa:
    def __init__(self, env, tiling):
        self.env = env
        self.tiling = tiling

    def train(self, iterations, n, epsilon, alpha, gamma=1):
        w = np.zeros(self.tiling.get_size())
        sum_rewards = []

        for _ in range(iterations):
            states = []
            rewards = [0]
            sum_reward = 0
            actions = []
            state, _ = self.env.reset()
            states.append(state)
            action = self.get_epsilon_greedy_action(epsilon, state, w)
            actions.append(action)

            T = float('inf')
            t = 0
            while True:
                if t < T:
                    next_state, reward, terminated, _, _ = self.env.step(action)
                    rewards.append(reward)
                    sum_reward += reward
                    states.append(next_state)

                    if terminated:
                        T = t + 1
                    else:
                        action = self.get_epsilon_greedy_action(epsilon, next_state, w)
                        actions.append(action)

                tau = t - n + 1
                if tau >= 0:
                    g = 0.0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        g = g + pow(gamma, i - tau - 1) * rewards[i]

                    if tau + n < T:
                        g = g + pow(gamma, n) * self.q_hat(states[tau + n], actions[tau + n], w)

                    delta_w = alpha * (g - self.q_hat(states[tau], actions[tau], w))
                    self.update_weights(delta_w, states[tau], actions[tau], w)

                if tau == T - 1:
                    break

                t += 1

            sum_rewards.append(sum_reward)

        return sum_rewards

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
