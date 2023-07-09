import numpy as np


class SarsaLambda:
    def __init__(self, env, get_binary_feature_indices, w_dimension):
        self.get_binary_feature_indices = get_binary_feature_indices
        self.env = env
        self.d = w_dimension

    def train(self, iterations, epsilon, alpha, trace_decay_rate, gamma=1.0):
        w = np.zeros(self.d)
        sum_rewards = []

        for _ in range(iterations):
            state, _ = self.env.reset()
            action = self.get_epsilon_greedy_action(epsilon, state, w)
            sum_reward = 0
            z = np.zeros(self.d)
            while True:
                next_state, reward, terminated, _, _ = self.env.step(action)
                sum_reward += reward

                delta = reward
                features = self.get_binary_feature_indices(state, [action])
                for i in features:
                    delta = delta - w[i]
                    z[i] = z[i] + 1   # accumulating traces
                    # z[i] = 1        # replacing traces

                if terminated:
                    w = w + alpha * delta * z
                    break

                next_action = self.get_epsilon_greedy_action(epsilon, next_state, w)
                next_features = self.get_binary_feature_indices(next_state, [next_action])
                for i in next_features:
                    delta = delta + gamma * w[i]

                w = w + alpha * delta * z
                z = gamma * trace_decay_rate * z

                state = next_state
                action = next_action

            sum_rewards.append(sum_reward)

        return sum_rewards

    def get_epsilon_greedy_action(self, epsilon, state, w):
        if np.random.random() >= epsilon:
            q = float("-inf")
            max_action = self.env.action_space.start
            for action in range(self.env.action_space.n):
                features = self.get_binary_feature_indices(state, [action])
                q_new = w.take(features).sum()
                if q_new > q:
                    q = q_new
                    max_action = action

            return max_action
        else:
            return self.env.action_space.sample()
