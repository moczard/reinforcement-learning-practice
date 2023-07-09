import numpy as np


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


class BanditTestBed:
    def __init__(self, k, means, variances):
        self.k = k
        self.bandits = list(zip(means, variances))

    def sample(self, action):
        return np.random.normal(self.bandits[action][0], self.bandits[action][1])

    def epsilon_greedy_bandit(self, epsilon, alpha=None):
        q_values = np.zeros(self.k)
        n = np.zeros(self.k)
        avg_reward = 0
        avg_rewards = [0]

        for i in range(1000):
            if np.random.random() < epsilon:
                action = np.random.randint(self.k)
            else:
                action = q_values.argmax()

            reward = self.sample(action)
            avg_reward += (1 / (i + 1)) * (reward - avg_reward)
            avg_rewards.append(avg_reward)

            n[action] += 1

            step_size = 1 / n[action] if alpha is None else alpha
            q_values[action] = q_values[action] + step_size * (reward - q_values[action])

        return avg_rewards

    def optimistic_init_greedy_bandit(self, alpha, init_value):
        q_values = np.full(self.k, init_value)
        avg_reward = 0
        avg_rewards = [0]

        for i in range(1000):
            action = q_values.argmax()

            reward = self.sample(action)
            avg_reward += (1 / (i + 1)) * (reward - avg_reward)
            avg_rewards.append(avg_reward)

            q_values[action] = q_values[action] + alpha * (reward - q_values[action])

        return avg_rewards

    def upper_confidence_bound_bandit(self, alpha, c):
        q_values = np.zeros(self.k)
        n = np.zeros(self.k)
        avg_reward = 0
        avg_rewards = [0]

        for i in range(1000):
            not_selected_bandits = np.where(n == 0)
            if len(not_selected_bandits[0]) > 0:
                action = np.random.choice(not_selected_bandits[0])
            else:
                action = (q_values + c * np.sqrt(np.log(i) / n)).argmax()

            reward = self.sample(action)
            avg_reward += (1 / (i + 1)) * (reward - avg_reward)
            avg_rewards.append(avg_reward)

            n[action] += 1

            q_values[action] = q_values[action] + alpha * (reward - q_values[action])

        return avg_rewards

    def non_stationary_epsilon_greedy_bandit(self, epsilon, alpha):
        q_values_constant_alpha = np.zeros(self.k)
        q_values = np.zeros(self.k)
        n = np.zeros(self.k)

        avg_reward = 0
        avg_rewards = [0]
        avg_reward_constant_alpha = 0
        avg_rewards_constant_alpha = [0]

        for i in range(10000):
            if np.random.random() < epsilon:
                action = np.random.randint(self.k)
                action_constant_alpha = action
            else:
                action = q_values.argmax()
                action_constant_alpha = q_values_constant_alpha.argmax()

            reward = self.sample(action)
            reward_constant_alpha = self.sample(action_constant_alpha)
            avg_reward += (1 / (i + 1)) * (reward - avg_reward)
            avg_rewards.append(avg_reward)
            avg_reward_constant_alpha += (1 / (i + 1)) * (reward_constant_alpha - avg_reward_constant_alpha)
            avg_rewards_constant_alpha.append(avg_reward_constant_alpha)

            n[action] += 1
            q_values[action] = q_values[action] + (1 / n[action]) * (reward - q_values[action])
            q_values_constant_alpha[action_constant_alpha] = \
                q_values_constant_alpha[action_constant_alpha] + \
                alpha * (reward_constant_alpha - q_values_constant_alpha[action_constant_alpha])

            for k in range(self.k):
                self.bandits[k] = (self.bandits[k][0] + np.random.normal(0, 0.01), 1)

        return avg_rewards, avg_rewards_constant_alpha

    def gradient_bandit(self, alpha):
        h = np.zeros(self.k)
        avg_reward = 0
        avg_rewards = [0]

        for i in range(1000):
            action = h.argmax()
            reward = self.sample(action)

            probs = softmax(h)
            for k in range(self.k):
                if k == action:
                    h[k] = h[k] + alpha * (reward - avg_reward) * (1 - probs[k])
                else:
                    h[k] = h[k] - alpha * (reward - avg_reward) * probs[k]

            avg_reward += (1 / (i + 1)) * (reward - avg_reward)
            avg_rewards.append(avg_reward)

        return avg_rewards
