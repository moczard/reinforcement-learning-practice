from matplotlib import pyplot as plt

from environments.dyna_maze import DynaMaze
from planning_and_learning.dyna_q import dyna_q


def compare_planning_dyna_q():
    env = DynaMaze()
    _, step_counter_1 = dyna_q(env, 50, 1)
    _, step_counter_5 = dyna_q(env, 50, 5)
    _, step_counter_50 = dyna_q(env, 50, 50)
    plt.plot(step_counter_1[1:], 'b')
    plt.plot(step_counter_5[1:], 'g')
    plt.plot(step_counter_50[1:], 'r')
    plt.ylabel('Steps per episode')
    plt.xlabel('Episodes')
    plt.show()
