from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from environments.cliff_walking import CliffWalking
from environments.maximization_bias_mdp import MaximizationBiasMDP
from environments.random_walk import RandomWalk
from environments.windy_grid_world import WindyGridWorld
from temporal_difference.expected_sarsa import expected_sarsa
from temporal_difference.q_learning import q_learning, q_learning_max_bias, double_q_learning_max_bias
from temporal_difference.sarsa import sarsa
from temporal_difference.td_0_prediction import td_0_prediction


def random_walk_td_prediction():
    random_walk_env = RandomWalk(7)
    state_values = td_0_prediction(random_walk_env, 100, random_walk_env.policy, 0.1)
    plt.plot(['A', 'B', 'C', 'D', 'E'], [state_values[i] for i in range(1, 6)], 'b')
    plt.plot(['A', 'B', 'C', 'D', 'E'], random_walk_env.true_values, 'g')
    plt.ylabel('Value')
    plt.xlabel('State')
    plt.show()


def windy_grid_world_sarsa():
    windy_grid_world = WindyGridWorld()
    q_values, step_counter, _ = sarsa(windy_grid_world, 170, 0.1, 0.5)
    plt.plot(step_counter, 'b')
    plt.ylabel('Episodes')
    plt.xlabel('Time steps')
    plt.show()


def cliff_walking_compare_td_algorithms():
    cliff_waling = CliffWalking()
    q_sums = []
    sarsa_sums = []
    expected_sarsa_sums = []
    for i in range(200):
        _, _, sum_rewards_q_learning = q_learning(cliff_waling, 100, 0.1, 0.5)
        _, _, sum_rewards_sarsa = sarsa(cliff_waling, 100, 0.1, 0.5)
        _, _, sum_rewards_expected_sarsa = expected_sarsa(cliff_waling, 100, 0.1, 0.5)
        q_sums.append(sum_rewards_q_learning)
        sarsa_sums.append(sum_rewards_sarsa)
        expected_sarsa_sums.append(sum_rewards_expected_sarsa)

    avg_q = np.array(q_sums).mean(axis=0)
    avg_sarsa = np.array(sarsa_sums).mean(axis=0)
    avg_expected_sarsa = np.array(expected_sarsa_sums).mean(axis=0)

    plt.plot(avg_q, 'b')
    plt.plot(avg_sarsa, 'g')
    plt.plot(avg_expected_sarsa, 'r')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.ylim(bottom=-100)
    plt.show()


def maximization_bias_example():
    env = MaximizationBiasMDP()
    initial_q_values = defaultdict(lambda: np.zeros(env.n_actions))
    initial_q_values[1] = np.zeros(10)

    q_percentages = []
    double_q_percentages = []

    for i in range(1000):
        left_action_percentage_q = q_learning_max_bias(env, 300)
        left_action_percentage_double_q = double_q_learning_max_bias(env, 300)
        q_percentages.append(left_action_percentage_q)
        double_q_percentages.append(left_action_percentage_double_q)

    avg_q = np.array(q_percentages).mean(axis=0)
    avg_double_q = np.array(double_q_percentages).mean(axis=0)

    plt.plot(avg_q, 'r')
    plt.plot(avg_double_q, 'g')

    plt.ylabel('% Left actions from A')
    plt.xlabel('Episodes')
    plt.show()
