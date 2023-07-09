import numpy as np


def play_episode(env, q_values):
    sum_reward = 0
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = q_values[state].argmax()
        next_state, reward, terminated, truncated, _ = env.step(action)
        sum_reward += reward
        state = next_state

    return sum_reward


def generate_episode(env, policy):
    episode = []
    state, _ = env.reset()
    terminated = False
    while not terminated:
        action = policy(state)
        next_state, reward, terminated, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    return episode


def generate_episode_epsilon_greedy(env, epsilon, q_values):
    episode = []
    state, _ = env.reset()
    terminated = False
    while not terminated:
        if np.random.random() >= epsilon:
            action = q_values[state].argmax()
        else:
            action = env.action_space.sample()
        next_state, reward, terminated, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    return episode


def get_prob_epsilon_greedy(state, action, epsilon, q_values, n_actions):
    max_action = q_values[state].argmax()
    if action == max_action:
        return 1 - epsilon + (epsilon / n_actions)
    else:
        return epsilon / n_actions


def get_probs_epsilon_greedy(state, epsilon, q_values, n_actions):
    max_action = q_values[state].argmax()
    probs = np.full(n_actions, epsilon / n_actions)
    probs[max_action] += 1 - epsilon

    return probs


def get_epsilon_greedy_action(state, epsilon, n_actions, q_values):
    if np.random.random() >= epsilon:
        return q_values[state].argmax()
    else:
        return np.random.randint(n_actions)


def get_epsilon_greedy_action_random_ties(epsilon, possible_actions, action_values):
    if np.random.random() >= epsilon:
        return np.random.choice(np.flatnonzero(action_values == action_values.max()))
    else:
        return np.random.choice(possible_actions)
