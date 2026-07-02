from mushroom_rl.environments import FiniteMDP
import numpy as np

class SelfDefineMDP(FiniteMDP):
    def __init__(self, environment:np.ndarray, gamma=0.99, horizon=100, prob=0.8):
        '''
        self define MDP class, which allows users to define their own MDP environment.
        environment(np.ndarray): a 2D numpy array, where each element represents a 
        state in the MDP.
        gamma(float): discount factor for the MDP.
        horizon(int): the maximum number of steps in an episode.
        prob(float): the probability of taking the intended action.
        '''
        self.environment = environment
        self.shape = environment.shape
        assert len(self.shape) == 2, "The environment must be a 2D array."
        self.prob = prob
        self.directions = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        self.cell_list = self._parse_cell_list()

        p = self.compute_probabilities(self.environment, self.cell_list, self.prob)
        r = self.compute_rewards(self.environment, self.cell_list)
        mu = self.initialize_state_distribution(self.cell_list, self.environment)
        # call the super class
        super().__init__(p, r, mu, gamma, horizon)
        self.reset()

    def _parse_cell_list(self):
        return
    
    def compute_probabilities(self, environment, cell_list, prob):
        return
    
    def compute_rewards(self, environment, cell_list):
        return
    
    def initialize_state_distribution(self, cell_list, environment):
        return