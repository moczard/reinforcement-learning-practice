from collections import defaultdict

from utils.utils import generate_episode


def mc_prediction(env, iterations, policy, gamma=1):
    state_values = defaultdict()
    returns = defaultdict(lambda: [])

    for i in range(iterations):
        episode = generate_episode(env, policy)
        states, actions, rewards = zip(*episode)
        states = list(states)
        g = 0
        for t in reversed(range(len(episode))):
            g = gamma * g + rewards[t]
            if states[t] not in states[0:t]:
                returns[states[t]].append(g)
                state_values[states[t]] = sum(returns[states[t]]) / len(returns[states[t]])

    return state_values
