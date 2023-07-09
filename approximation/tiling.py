import numpy as np

from utils.tiles3 import tiles, IHT


class Tiling:
    def __init__(self, state_sizes, max_size, number_of_tilings, tiling_size):
        self.max_size = max_size
        self.number_of_tilings = number_of_tilings

        self.iht = IHT(max_size)

        tiling_sizes = np.full(len(state_sizes), tiling_size)
        self.scale_factors = np.divide(tiling_sizes, state_sizes)

    def get_tiles(self, state, actions):
        return tiles(self.iht, self.number_of_tilings, np.multiply(state, self.scale_factors), actions)

    def get_size(self):
        return self.max_size
