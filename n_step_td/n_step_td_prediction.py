import numpy as np


def n_step_td_prediction(env, iterations, n, policy, alpha, gamma=1):
    state_values = np.zeros(env.n_states)

    for _ in range(iterations):
        states = []
        rewards = [0]
        state = env.reset()
        states.append(state)

        T = float('inf')
        t = 0
        while True:

            if t < T:
                action = policy(state)
                next_state, reward, done = env.step(action)
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
                    g = g + pow(gamma, n) * state_values[states[tau + n]]
                state_values[states[tau]] = state_values[states[tau]] + alpha * (g - state_values[states[tau]])

            if tau == T - 1:
                break

            t += 1

    return state_values, env.get_rmse(state_values)
