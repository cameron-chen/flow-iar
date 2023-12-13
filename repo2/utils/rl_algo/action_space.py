#!/usr/bin/env python
# -*- coding: utf-8 -*-

# [reference] Use and modified code in https://github.com/ChangyWen/wolpertinger_ddpg/blob/master/src/action_space.py

import itertools

import numpy as np
import pyflann
from einops import rearrange

"""
    This class represents a n-dimensional unit cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.
"""


class Space:

    def __init__(self, low, high, points, mode='medium'):

        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)
        self._space_low = -1
        self._space_high = 1
        self._k = (self._space_high - self._space_low) / self._range
        self.__space = init_uniform_space([self._space_low] * self._dimensions,
                                          [self._space_high] * self._dimensions,
                                          points)
        self._flann = pyflann.FLANN()
        self.mode = mode # 'medium' or 'slow'
        self.rebuild_flann()

    def rebuild_flann(self):
        if self.mode == 'medium':
            self._index = self._flann.build_index(self.__space, algorithm='kdtree')
        elif self.mode == 'slow':
            self._index = self._flann.build_index(self.__space, algorithm='kmeans', branching=16)
        else:
            raise ValueError('flann_mode must be medium or slow')

    def search_point(
        self, 
        point: np.ndarray, 
        k: int
    ):
        """Search for the k nearest neighbors of the input point.

        Args:
            point (np.ndarray): The input point in the original space.
            k (int): The number of nearest neighbors to return.
        
        Returns:
            knns (np.ndarray): The k nearest neighbors of the input point in the scaled space. 
            p_out (np.ndarray): The k nearest neighbors of the input point in the original space.

        Note: 
            Original space: [low, high]
            Scaled space: [space_low, space_high]
        """
        p_in = point
        if not isinstance(point, np.ndarray):
            p_in = np.array([p_in]).astype(np.float64)
        p_in = self.import_point(point)
        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self.__space[search_res].astype(np.float32)

        if k == 1:
            knns = rearrange(knns, 'n ... -> n () ...')
        p_out = []
        for p in knns:
            p_out.append(self.export_point(p)) 
        return knns, np.array(p_out)

    def import_point(self, point):
        """Convert a point in the original space to the scaled space."""
        return self._space_low + self._k * (point - self._low)

    def export_point(self, point):
        """Convert a point in the scaled space to the original space."""
        return self._low + (point - self._space_low) / self._k

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]


class Discrete_space(Space):
    """
        Discrete action space with n actions (the integers in the range [0, n))
        1, 2, ..., n-1, n

        In gym: 'Discrete' object has no attribute 'high'
    """

    def __init__(self, n, mode='medium'):  # n: the number of the discrete actions
        super().__init__([0], [n-1], n, mode=mode)

    def export_point(self, point):
        ex_p = np.round(super().export_point(point)).astype(int)
        return ex_p

class Multi_discrete_space(Space):
    """
        Multi discrete action space 

    """

    def __init__(self, action_low, action_high, n, mode='medium'):  # n: the number of the discrete actions
        super().__init__(action_low, action_high, n, mode=mode)

    def export_point(self, point):
        ex_p = np.round(super().export_point(point)).astype(int)
        return ex_p


def init_uniform_space(low, high, points):
    dims = len(low)
    # In Discrete situation, the action space is an one dimensional space, i.e., one row
    points_in_each_axis = round(points**(1 / dims))

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))

    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    # space: e.g., [[1], [2], ... ,[n-1]]
    return np.array(space)

