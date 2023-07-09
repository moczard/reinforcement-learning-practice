import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from approximation.tiling import Tiling
from eligibility_traces.sarsa_lambda import SarsaLambda
from eligibility_traces.semi_gradient_td_lambda import SemiGradientTDLambda
from eligibility_traces.true_online_sarsa_lambda import TrueOnlineSarsaLambda
from eligibility_traces.true_online_td_lambda import TrueOnlineTDLambda
from environments.random_walk import RandomWalk


def td_lambda_prediction():
    env = RandomWalk(21)
    semi_gradient_td_lambda_agent = SemiGradientTDLambda(env, random_walk_v_hat, random_walk_delta_v_hat, 19)
    avg_errors_04 = get_random_walk_error(env, semi_gradient_td_lambda_agent, 0.4)

    x_scale = [j / 100 for j in range(100)]
    plt.plot(x_scale, avg_errors_04, 'g')
    plt.ylabel('RMSE')
    plt.xlabel('Alpha')
    plt.ylim(top=0.55)
    plt.show()


def true_online_td_lambda_prediction():
    env = RandomWalk(21)
    true_online_td_lambda_agent = TrueOnlineTDLambda(env, random_walk_delta_v_hat, 19)
    avg_errors_04 = get_random_walk_error(env, true_online_td_lambda_agent, 0.4)

    x_scale = [j / 100 for j in range(100)]
    plt.plot(x_scale, avg_errors_04, 'g')
    plt.ylabel('RMSE')
    plt.xlabel('Alpha')
    plt.ylim(top=0.55)
    plt.show()


def mountain_car_sarsa_lambda():
    env = gymnasium.make("MountainCar-v0")
    tiling = Tiling([1.2 + 0.6, 0.07 + 0.07], 4096, 8, 8.0)
    sarsa_lambda = SarsaLambda(env, tiling.get_tiles, tiling.get_size())
    sum_rewards = sarsa_lambda.train(1000, 0.01, 0.2 / 8.0, 0.7, 1.0)
    plt.plot(sum_rewards, 'g')
    plt.ylabel('rewards')
    plt.xlabel('Episodes')
    plt.show()


def mountain_car_true_online_sarsa():
    env = gymnasium.make("MountainCar-v0")
    tiling = Tiling([1.2 + 0.6, 0.07 + 0.07], 4096, 8, 8.0)
    true_online_sarsa = TrueOnlineSarsaLambda(env, tiling)
    rewards, avg_rewards = true_online_sarsa.train(1000, 0.01, 0.2 / 8.0, 0.92, 1.0)
    plt.plot(rewards, 'r')
    plt.ylabel('rewards')
    plt.xlabel('Episodes')
    plt.show()


def get_random_walk_error(env, agent, trace_decay_rate):
    run_errors = np.empty((100, 100))
    for i in range(100):
        for j in range(100):
            w = agent.train(10, env.policy, (j + 1) / 100, trace_decay_rate)
            state_values = [random_walk_v_hat(s, w) for s in range(21)]
            rmse = env.get_rmse(np.array(state_values))

            run_errors[i][j] = rmse
    avg_errors = run_errors.mean(axis=0)
    return avg_errors


def random_walk_v_hat(state, w):
    if state == 0 or state == 20:
        return 0

    return w[state - 1]


def random_walk_delta_v_hat(state, w=None):
    delta_v = np.zeros(19)
    if state == 0 or state == 20:
        return delta_v

    delta_v[state - 1] = 1
    return delta_v
