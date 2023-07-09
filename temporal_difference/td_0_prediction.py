from collections import defaultdict


def td_0_prediction(env, iterations, policy, alpha=0.1, gamma=1.0):
    state_values = defaultdict(lambda: 0.0)
    for i in range(iterations):
        state = env.reset()
        terminated = False
        while not terminated:
            next_state, reward, terminated = env.step(policy(state))
            state_values[state] = state_values[state] + alpha * (reward + gamma * state_values[next_state] - state_values[state])
            state = next_state

    return state_values
