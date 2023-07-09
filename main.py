import gymnasium

from approximation.approximation_examples import random_walk_prediction, mountain_car_with_episodic_semi_gradient_sarsa, \
    random_walk_least_squares_td
from dynamic_programming.dynamic_programming_examples import grid_world_policy_evaluation,\
    grid_world_policy_iteration, car_rental_policy_iteration, gambler_value_iteration
from eligibility_traces.eligibility_traces_examples import td_lambda_prediction, true_online_td_lambda_prediction, \
    mountain_car_sarsa_lambda, mountain_car_true_online_sarsa
from k_armed_bandit.bandit_problems import compare_epsilon_greedy_bandits, \
    compare_epsilon_greedy_and_optimistic_init_bandits, compare_epsilon_greedy_and_ucb_bandits, \
    compare_non_stationary_bandits, compare_epsilon_greedy_and_gradient_bandits
from monte_carlo.monte_carlo_examples import black_jack_mc_prediction, black_jack_mc_control, \
    black_jack_off_policy_mc_prediction, black_jack_off_policy_mc_control
from n_step_td.n_step_td_examples import compare_n_step_td_predictions, off_policy_n_step_sarsa_taxi,\
    n_step_sarsa_taxi, n_step_tree_backup_taxi
from planning_and_learning.planning_and_learning_examples import compare_planning_dyna_q
from policy_gradient_methods.reinforce import Reinforce
from temporal_difference.td_examples import random_walk_td_prediction, windy_grid_world_sarsa, \
    cliff_walking_compare_td_algorithms, maximization_bias_example


def cliff_walking_q_learning_vs_sarsa():
    pass


if __name__ == '__main__':
    # Bandit problems
    # compare_epsilon_greedy_bandits()
    # compare_epsilon_greedy_and_optimistic_init_bandits()
    # compare_epsilon_greedy_and_ucb_bandits()
    # compare_non_stationary_bandits()
    # compare_epsilon_greedy_and_gradient_bandits()
    #
    # Dynamic Programming
    # grid_world_policy_evaluation()
    # grid_world_policy_iteration()
    # car_rental_policy_iteration()
    # gambler_value_iteration()
    #
    # Monte Carlo methods
    # black_jack_mc_prediction()
    # black_jack_mc_control()
    # black_jack_off_policy_mc_prediction()
    # black_jack_off_policy_mc_control()
    #
    # Temporal-Difference learning
    # random_walk_td_prediction()
    # windy_grid_world_sarsa()
    # cliff_walking_q_learning_vs_sarsa()
    # cliff_walking_compare_td_algorithms()
    # maximization_bias_example()
    #
    # N-step TD methods
    # compare_n_step_td_predictions()
    # n_step_sarsa_taxi()
    # off_policy_n_step_sarsa_taxi()
    # n_step_tree_backup_taxi()

    # Planning and Learning
    # compare_planning_dyna_q()

    # Approximation
    # random_walk_prediction()
    # mountain_car_with_episodic_semi_gradient_sarsa()
    # random_walk_least_squares_td()

    # Eligibility Traces
    # td_lambda_prediction()
    # true_online_td_lambda_prediction()
    # mountain_car_sarsa_lambda()
    # mountain_car_true_online_sarsa()

    # env = gymnasium.make("LunarLander-v2")
    # tiling = Tiling([180.0, 180.0, 10.0, 10.0, 3.1415927 + 3.1415927, 10, 1, 1], 8096, 9, 10.0)
    # true_online_sarsa = TrueOnlineSarsaLambda(env, tiling)
    # rewards, avg_rewards = true_online_sarsa.train(1000, 0.01, 0.2 / 8.0, 0.7, 1.0)
    # plt.plot(rewards, 'g')
    # plt.plot(avg_rewards, 'r')
    # plt.ylabel('rewards')
    # plt.xlabel('Episodes')
    # plt.show()
    env = gymnasium.make("CartPole-v1")
    rein_agent = Reinforce(env)
    rein_agent.train(2000, 0.001, 0.99)

