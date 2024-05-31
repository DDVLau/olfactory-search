"""
    Reference: 
     - Paper: 
     - Code: 
"""

from abc import abstractmethod
from collections import namedtuple

import numpy as np
import scipy

import pygame
import gymnasium as gym
from gymnasium import spaces


class OlfactoryEnvContinous(gym.Env):
    """
    Base class for continuous olfactory environment.
    """
    def __init__(self, config):
        super(OlfactoryEnvContinous, self).__init__()
        self.config = config

        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([self.config.width, self.config.height]), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([self.config.width, self.config.height]), dtype=np.float32)

        self.viewer = None

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        pass

    def set_state(self, state):
        pass

    def get_state(self):
        pass

    def get_reward(self):
        pass

    def get_done(self):
        pass

    def get_info(self):
        pass

    def get_distance(self, state1, state2):
        return np.linalg.norm(np.array(state1) - np.array(state2))

    def get_angle(self, state1, state2):
        return np.arctan2(state2[1] - state1[1], state2[0] - state1[0])

    def get_distance_angle(self, state1, state2):
        return self.get_distance(state1, state2), self.get_angle(state1, state2)