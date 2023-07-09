import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from environments.random_walk import RandomWalk
from n_step_td.n_step_sarsa import n_step_sarsa
from n_step_td.n_step_td_prediction import n_step_td_prediction
from n_step_td.n_step_tree_backup import n_step_tree_backup
from n_step_td.off_policy_n_step_sarsa import off_policy_n_step_sarsa
from utils.utils import play_episode


def compare_n_step_td_predictions():
    env = RandomWalk(21)
    avg_errors_1 = get_random_walk_error_for_n_step_td(env, 1)
    avg_errors_2 = get_random_walk_error_for_n_step_td(env, 2)
    avg_errors_4 = get_random_walk_error_for_n_step_td(env, 4)
    avg_errors_8 = get_random_walk_error_for_n_step_td(env, 8)
    avg_errors_16 = get_random_walk_error_for_n_step_td(env, 16)
    avg_errors_32 = get_random_walk_error_for_n_step_td(env, 32)
    x_scale = [j / 100 for j in range(100)]
    plt.plot(x_scale, avg_errors_1, 'r')
    plt.plot(x_scale, avg_errors_2, 'g')
    plt.plot(x_scale, avg_errors_4, 'b')
    plt.plot(x_scale, avg_errors_8, 'k')
    plt.plot(x_scale, avg_errors_16, 'm')
    plt.plot(x_scale, avg_errors_32, 'c')
    plt.ylabel('RMSE')
    plt.xlabel('Alpha')
    plt.ylim(top=0.55)
    plt.show()


def n_step_sarsa_taxi():
    taxi_env = gymnasium.make("Taxi-v3")
    _, rewards = n_step_sarsa(taxi_env, 1000, 4, 0.01, 0.1)
    print(np.array(rewards[-100:]).mean())
    plt.plot(rewards, 'r')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.ylim(bottom=-500)
    plt.show()


def off_policy_n_step_sarsa_taxi():
    taxi_env = gymnasium.make("Taxi-v3")
    q_values, rewards = off_policy_n_step_sarsa(taxi_env, 1000, 4, 0.01, 0.1)
    print(np.array(rewards[-100:]).mean())

    taxi_env_render = gymnasium.make("Taxi-v3", render_mode="human")
    play_episode(taxi_env_render, q_values)


def n_step_tree_backup_taxi():
    taxi_env = gymnasium.make("Taxi-v3")
    q_values, _ = n_step_tree_backup(taxi_env, 1000, 4, 0.01, 0.1)

    rewards = []
    for i in range(100):
        sum_reward = play_episode(taxi_env, q_values)
        rewards.append(sum_reward)

    print(np.array(rewards[-100:]).mean())
    plt.plot(rewards, 'r')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.show()


def get_random_walk_error_for_n_step_td(env, n):
    run_errors = np.empty((100, 100))
    for i in range(100):
        for j in range(100):
            _, rmse = n_step_td_prediction(env, 10, n, env.policy, (j + 1) / 100)
            run_errors[i][j] = rmse
    avg_errors = run_errors.mean(axis=0)
    return avg_errors
