import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from approximation.episodic_semi_gradient_n_step_sarsa import EpisodicSemiGradientNStepSarsa
from approximation.episodic_semi_gradient_sarsa import EpisodicSemiGradientSarsa
from approximation.gradient_mc_prediction import GradientMCPrediction
from approximation.least_squares_td import LeastSquaresTD
from approximation.n_step_semi_gradient_td_prediction import NStepSemiGradientTDPrediction
from approximation.semi_gradient_td_0 import SemiGradientTDPrediction
from approximation.tiling import Tiling
from environments.thousand_state_random_walk import ThousandStateRandomWalk


def random_walk_prediction():
    random_walk_env = ThousandStateRandomWalk()

    gradient_mc_agent = GradientMCPrediction(random_walk_env, random_walk_v_hat, random_walk_delta_v_hat, np.zeros(10))
    semi_gradient_td_0 = SemiGradientTDPrediction(random_walk_env, random_walk_v_hat, random_walk_delta_v_hat, np.zeros(10))
    n_step_semi_gradient_td = NStepSemiGradientTDPrediction(random_walk_env, random_walk_v_hat, random_walk_delta_v_hat, np.zeros(10))

    w_mc = gradient_mc_agent.train(100000, random_walk_env.policy, 0.00002)
    w_td = semi_gradient_td_0.train(100000, random_walk_env.policy, 0.001)
    w_n_step_td = n_step_semi_gradient_td.train(100000, random_walk_env.policy, 4, 0.001)

    state_values_mc = []
    state_values_td = []
    state_values_n_step_td = []
    for s in range(1, 1001):
        state_values_mc.append(random_walk_v_hat(s, w_mc))
        state_values_td.append(random_walk_v_hat(s, w_td))
        state_values_n_step_td.append(random_walk_v_hat(s, w_n_step_td))

    plt.plot(state_values_mc, 'b')
    plt.plot(state_values_td, 'g')
    plt.plot(state_values_n_step_td, 'k')
    plt.plot([i / 500 for i in range(-500, 500)], 'r')
    plt.ylabel('Value')
    plt.xlabel('State')
    plt.show()


def mountain_car_with_episodic_semi_gradient_sarsa():
    env = gymnasium.make("MountainCar-v0")
    tiling = Tiling([1.2 + 0.6, 0.07 + 0.07], 4096, 8, 8.0)
    episodic_semi_gradient_sarsa = EpisodicSemiGradientSarsa(env, tiling)

    tiling_n_step = Tiling([1.2 + 0.6, 0.07 + 0.07], 4096, 8, 8.0)
    episodic_semi_gradient_n_step_sarsa = EpisodicSemiGradientNStepSarsa(env, tiling_n_step)

    sarsa_sums = []
    n_step_sarsa_sums = []
    for i in range(20):
        rewards = episodic_semi_gradient_sarsa.train(400, 0.01, 0.5 / 8.0)
        rewards_n_step = episodic_semi_gradient_n_step_sarsa.train(400, 8, 0.01, 0.3 / 8.0)
        sarsa_sums.append(rewards)
        n_step_sarsa_sums.append(rewards_n_step)

    avg_sarsa = np.array(sarsa_sums).mean(axis=0)
    avg_n_step_sarsa = np.array(n_step_sarsa_sums).mean(axis=0)

    plt.plot(-1.0 * np.array(avg_sarsa), 'r')
    plt.plot(-1.0 * np.array(avg_n_step_sarsa), 'b')
    plt.ylabel('rewards')
    plt.xlabel('Episodes')
    plt.ylim(top=1000)
    plt.show()


def random_walk_least_squares_td():
    random_walk_env = ThousandStateRandomWalk()

    least_squares_td_agent = LeastSquaresTD(random_walk_env, random_walk_feature_representation, np.zeros(10))
    w_least_squares_td = least_squares_td_agent.train(10000, random_walk_env.policy, 0.01)

    state_values_least_squares_td = []
    for s in range(1, 1001):
        state_values_least_squares_td.append(random_walk_v_hat(s, w_least_squares_td))

    plt.plot(state_values_least_squares_td, 'b')
    plt.plot([i / 500 for i in range(-500, 500)], 'r')
    plt.ylabel('Value')
    plt.xlabel('State')
    plt.show()


def random_walk_v_hat(state, w):
    if state == 0 or state == 1001:
        return 0

    return w[int((state - 1) / 100)]


def random_walk_delta_v_hat(state, w):
    delta_v = np.zeros(10)
    delta_v[int((state - 1) / 100)] = 1
    return delta_v


def random_walk_feature_representation(state):
    x = np.zeros(10)
    if state == 0 or state == 1001:
        return x

    x[int((state - 1) / 100)] = 1
    return x
