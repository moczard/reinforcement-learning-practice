import numpy as np
from matplotlib import pyplot as plt

from k_armed_bandit.badit_test_bed import BanditTestBed


def run_epsilon_greedy(num_of_runs, epsilon, alpha=None):
    avg_run_rewards = []
    for run_number in range(num_of_runs):
        test_bed = get_test_bed()

        avg_rewards = test_bed.epsilon_greedy_bandit(epsilon, alpha)
        avg_run_rewards.append(avg_rewards)

    return np.array(avg_run_rewards).mean(axis=0)


def run_optimistic_init_greedy(num_of_runs, alpha):
    avg_run_rewards = []
    for run_number in range(num_of_runs):
        test_bed = get_test_bed()

        avg_rewards = test_bed.optimistic_init_greedy_bandit(alpha, 5.0)
        avg_run_rewards.append(avg_rewards)

    return np.array(avg_run_rewards).mean(axis=0)


def run_upper_confidence_bound(num_of_runs, alpha, c):
    avg_run_rewards = []
    for run_number in range(num_of_runs):
        test_bed = get_test_bed()

        avg_rewards = test_bed.upper_confidence_bound_bandit(alpha, c)
        avg_run_rewards.append(avg_rewards)

    return np.array(avg_run_rewards).mean(axis=0)


def run_non_stationary(num_of_runs, epsilon, alpha):
    avg_rewards_sample_averages = []
    avg_rewards_constant_alpha = []
    for run_number in range(num_of_runs):
        test_bed = get_test_bed()

        sample_avg_rewards, constant_alpha_rewards = test_bed.non_stationary_epsilon_greedy_bandit(epsilon, alpha)
        avg_rewards_sample_averages.append(sample_avg_rewards)
        avg_rewards_constant_alpha.append(constant_alpha_rewards)

    return np.array(avg_rewards_sample_averages).mean(axis=0), np.array(avg_rewards_constant_alpha).mean(axis=0)


def run_gradient_algorithm(num_of_runs, alpha):
    avg_run_rewards = []
    for run_number in range(num_of_runs):
        test_bed = get_test_bed()

        avg_rewards = test_bed.gradient_bandit(alpha)
        avg_run_rewards.append(avg_rewards)

    return np.array(avg_run_rewards).mean(axis=0)


def get_test_bed():
    means = np.random.normal(0, 1, 10)
    variances = np.ones(10)
    test_bed = BanditTestBed(10, means, variances)
    return test_bed


def compare_epsilon_greedy_bandits():
    avg_rewards1 = run_epsilon_greedy(2000, 0.1)
    avg_rewards2 = run_epsilon_greedy(2000, 0.01)
    avg_rewards3 = run_epsilon_greedy(2000, 0)

    plt.plot(avg_rewards1, 'b')
    plt.plot(avg_rewards2, 'r')
    plt.plot(avg_rewards3, 'g')
    plt.ylabel('Average rewards')
    plt.xlabel('Steps')
    plt.show()


def compare_epsilon_greedy_and_optimistic_init_bandits():
    avg_rewards_epsilon_greedy = run_epsilon_greedy(2000, 0.1, 0.1)
    avg_rewards_optimistic_init = run_optimistic_init_greedy(2000, 0.1)

    plt.plot(avg_rewards_epsilon_greedy, 'b')
    plt.plot(avg_rewards_optimistic_init, 'r')

    plt.ylabel('Average rewards')
    plt.xlabel('Steps')
    plt.show()


def compare_epsilon_greedy_and_ucb_bandits():
    avg_rewards_epsilon_greedy = run_epsilon_greedy(2000, 0.1, 0.1)
    avg_rewards_ucb = run_upper_confidence_bound(2000, 0.1, 1.5)

    plt.plot(avg_rewards_epsilon_greedy, 'b')
    plt.plot(avg_rewards_ucb, 'r')

    plt.ylabel('Average rewards')
    plt.xlabel('Steps')
    plt.show()


def compare_non_stationary_bandits():
    avg_rewards_sample_averages, avg_rewards_constant_alpha = run_non_stationary(2000, 0.1, 0.1)

    plt.plot(avg_rewards_sample_averages, 'b')
    plt.plot(avg_rewards_constant_alpha, 'r')
    plt.ylabel('Average rewards')
    plt.xlabel('Steps')
    plt.show()


def compare_epsilon_greedy_and_gradient_bandits():
    avg_rewards_epsilon_greedy = run_epsilon_greedy(2000, 0.1, 0.1)
    avg_rewards_gradient = run_gradient_algorithm(2000, 0.01)

    plt.plot(avg_rewards_epsilon_greedy, 'b')
    plt.plot(avg_rewards_gradient, 'r')

    plt.ylabel('Average rewards')
    plt.xlabel('Steps')
    plt.show()
