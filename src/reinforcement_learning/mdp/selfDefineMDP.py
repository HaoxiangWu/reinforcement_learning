from mushroom_rl.environments import FiniteMDP
from mushroom_rl.utils.viewer import Viewer
import numpy as np
from abc import ABC, abstractmethod

class SelfDefineMDP(FiniteMDP, ABC):
    def __init__(self, grid:np.ndarray, gamma=0.99, horizon=100, prob=0.8, **kwargs):
        '''
        self define MDP class, which allows users to define their own MDP environment.
        grid(np.ndarray): a 2D numpy array, where each element represents a 
        state in the MDP.
        gamma(float): discount factor for the MDP.
        horizon(int): the maximum number of steps in an episode.
        prob(float): the probability of taking the intended action.
        '''
        self.grid = grid
        self.shape = grid.shape
        assert len(self.shape) == 2, "The grid must be a 2D array."
        self.prob = prob
        self.directions = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        self.params = kwargs
        self.cell_list = self._parse_cell_list()
        self.viewer = Viewer(self.shape[0]+1, self.shape[1]+1, 500, 500)

        p = self.compute_probabilities(self.grid, self.cell_list, self.prob, **self.params)
        r = self.compute_rewards(self.grid, self.cell_list, **self.params)
        mu = self.initialize_state_distribution(self.cell_list, self.grid, **self.params)
        # call the super class
        super().__init__(p, r, mu, gamma, horizon)
        self.reset()

    @abstractmethod
    def _parse_cell_list(self):
        pass
    
    @abstractmethod
    def compute_probabilities(self, grid: np.ndarray, cell_list: list, prob: float, **kwargs):
        pass
    
    @abstractmethod
    def compute_rewards(self, grid: np.ndarray, cell_list: list, **kwargs):
        pass
    
    @abstractmethod
    def initialize_state_distribution(self, cell_list: list, grid: np.ndarray, **kwargs):
        pass
