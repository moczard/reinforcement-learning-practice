import numpy as np
from matplotlib import pyplot as plt

from dynamic_programming.policy_evaluation import policy_evaluation
from dynamic_programming.policy_iteration import PolicyIteration
from dynamic_programming.value_iteration import ValueIteration
from environments.car_rental import CarRental
from environments.gambler import GamblerEnv
from environments.grid_world import GridWorld


def grid_world_policy_evaluation():
    grid_env = GridWorld(4, 4, [0, 15], 1)
    random_policy = np.zeros((16, 4))
    random_policy.fill(0.25)
    s = policy_evaluation(grid_env, random_policy)
    print(np.reshape(s, (4, 4)))


def grid_world_policy_iteration():
    grid_env = GridWorld(4, 4, [0, 15], 1)
    policy_iteration_agent = PolicyIteration(grid_env, np.zeros(grid_env.n_states), 0.9)
    policy = policy_iteration_agent.policy_iteration()
    print(policy)


def car_rental_policy_iteration():
    car_rental_env = CarRental()
    start_policy = np.full(car_rental_env.n_states, 5)
    policy_iteration_agent = PolicyIteration(car_rental_env, start_policy, 0.9)
    policy = policy_iteration_agent.policy_iteration()

    reshaped_policy = np.array(policy - 5).reshape((21, 21))
    print(reshaped_policy)


def gambler_value_iteration():
    gambler_env = GamblerEnv(0.4)
    start_policy = np.zeros(gambler_env.n_states)
    value_iteration_agent = ValueIteration(gambler_env, start_policy)
    optimal_policy = value_iteration_agent.value_iteration()
    print(optimal_policy)
    plt.plot(optimal_policy, 'b')
    plt.ylabel('Policy')
    plt.xlabel('Capital')
    plt.show()
