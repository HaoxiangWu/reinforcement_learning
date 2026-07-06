from reinforcement_learning import SelfDefineMDP
import numpy as np
from copy import deepcopy
import pygame
pygame.init()
screen = pygame.display.set_mode((500, 500))
import cv2
import matplotlib.pyplot as plt


class StairToHeaven(SelfDefineMDP):
    def _make_grid(self, grid: np.ndarray):
        # Ensure the grid is a 2D numpy array
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError("Grid must be a 2D numpy array.")
        H, W = grid.shape
        # Create a copy of the grid to modify
        grid2=grid.copy()
        grid2[0][0]='.'
        grid2[np.where(grid2=='S')]='.'
        grid2[H//2][W//2]='.'
        grid_gather = np.array([grid, grid2])
        return grid_gather
        
    def _parse_cell_list(self):
        cell_list = []
        for s in range(self.shape[0]):
            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    if self.grid[s, i, j] != '#' and self.grid[s, i, j] != 'B':
                        cell_list.append([s, i, j])
        return cell_list
    
    def compute_probabilities(self, grid: np.ndarray, cell_list: list, prob: float, **kwargs):
        # Compute the transition probability matrix.
        c = np.array(cell_list)
        n_states = len(cell_list)
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        n_actions = len(directions)
        p = np.zeros((n_states, n_actions, n_states))
        collapse_prob = kwargs.get('collapse_prob', 0.05)  # Default collapse probability
        
        for i in range(len(c)):
            state = c[i]
            if grid[tuple(state)] in ['.', 'S']:
                for a in range(n_actions):
                    new_state = state[1:] + directions[a]
                    #if new state stays on the button
                    if (new_state == [0, 0, 0]).all():
                        new_state[0] = 1
                    
                    j = np.where((c == new_state).all(axis=1))[0]
                    if j.size > 0:
                        assert j.size == 1
                        p[i, a, i] = 1. - prob
                        p[i, a, j] = prob
                    # if new state hits the lava
                    else:
                        p[i, a, i] = 1.
                    
                    # Add collapse probability to the transition matrix
                    if new_state[0] == 1:
                        i_c = np.where((c == [0, state[1], state[2]]).all(axis=1))[0]
                        j_c = np.where((c == [0, new_state[1], new_state[2]]).all(axis=1))[0]
                        if i_c.size > 0:
                            assert i_c.size == 1
                            p[i, a, i_c] = collapse_prob * p[i, a, i]
                            p[i, a, i] *= 1 - collapse_prob
                        if j_c.size > 0:
                            assert j_c.size == 1
                            p[i, a, j_c] = collapse_prob * p[i, a, j]
                            p[i, a, j] *= 1 - collapse_prob
        return p
    
    def compute_rewards(self, grid:np.ndarray, cell_list: list, **kwargs):
        n_states = len(cell_list)
        c = np.array(cell_list)
        directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        n_actions = len(directions)
        r = np.ones((n_states, n_actions, n_states)) * -0.01
        def give_reward(poses:np.ndarray, rew:float):
            for current_state in poses:
                j = np.where((c == current_state).all(axis=1))[0]
                for a in range(n_actions):
                    prev_state = deepcopy(current_state)
                    prev_state[1:] -= directions[a]
                    i = np.where((c == prev_state).all(axis=1))[0]
                    if i.size > 0:
                        assert i.size == 1
                        if j.size < 1:
                            j = deepcopy(i)
                        r[i, a, j] = rew

        pose_G = np.argwhere(grid == 'G')
        give_reward(pose_G, 1.0)
        pose_L = np.argwhere(grid == '#')
        give_reward(pose_L, -0.1)
        return r
    
    def initialize_state_distribution(self, cell_list, grid, **kwargs):
        """
        Compute the initial states distribution.

        Args:
            grid_map (list): list containing the grid structure;
            cell_list (list): list of non-wall cells.

        Returns:
            The initial states distribution.

        """
        c = np.array(cell_list)
        n_states = len(c)
        # initialize intital state distribution
        mu = np.zeros(n_states)

        starts = np.argwhere(grid == 'S')
        for s in starts:
            i = np.where((c == s).all(axis=1))[0]
            # equal probability for all start states
            mu[i] = 1. / len(starts)

        return mu
    
    def render(self):
      _, H, W = self.grid.shape
      # create the grid
      for row in range(0, H+1):
        for col in range(0, W+1):
          self.viewer.line(np.array([col, 0]), np.array([col, H+1]))
          self.viewer.line(np.array([0, row]), np.array([W+1, row]))

      # get the stairway status and the agent pose
      s, agent_x, agent_y = self.cell_list[self._state[0]]
      for i in range(0, H):
        for j in range(0, W):
          # fill the cell with a gray square
          self.viewer.square(self.tf(i, j), 0, 1, color='gray')
          # fill with blue circle for the button
          if i == 0 and j == 0:
            self.viewer.circle(self.tf(i, j), 0.4, color='blue')
          # fill with red if lava
          if self.grid_map[s][i][j] == '#':
            self.viewer.square(self.tf(i, j), 0, 1, color='red')
          # fill with green for the goal
          if self.grid[s][i][j] == 'G':
            self.viewer.square(self.tf(i, j), 0, 1, color='green')
          # fill in with yellow arrow if agent
          if i == agent_x and j == agent_y:
            self.viewer.arrow_head(self.tf(i, j), 1, 0, color='yellow')

      # render the viewer
      pygame.display.flip()
      # visualize
      pygame.image.save(self.viewer.screen,'src/reinforcement_learning/pictures/image.png')
      img = cv2.imread('image.png')
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      plt.imshow(img)
      plt.show()
      plt.pause(0.1)
    
