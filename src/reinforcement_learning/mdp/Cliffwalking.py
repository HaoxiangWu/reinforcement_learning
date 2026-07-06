import numpy as np
from reinforcement_learning import SelfDefineMDP
import pygame
import cv2
import matplotlib.pyplot as plt


class CliffWalking(SelfDefineMDP):
    def _make_grid(self, grid: np.ndarray):
        """
        Create the grid for the CliffWalking environment.

        Args:
            grid (np.ndarray): 2D array representing the grid structure;
        Returns:
            The grid for the CliffWalking environment;
        """
        return grid

    def _parse_cell_list(self):
        cell_list = []
        H, W = self.shape
        for i in range(H):
            for j in range(W):
                if self.grid[i, j] in ['.', 'S', 'G']:
                    cell_list.append([i, j])
        return cell_list
    
    def compute_probabilities(self, grid: np.ndarray, cell_list: list, prob: float):
        """
        Compute the transition probability matrix.

        Args:
            grid (np.ndarray): 2D array representing the grid structure;
            cell_list (list): list of non-wall cells;
            prob (float): probability of success of an action.

        Returns:
            The transition probability matrix;

        """
        g = np.array(grid)
        c = np.array(cell_list)
        n_states = len(cell_list)
        p = np.zeros((n_states, 4, n_states))
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        # iterate over the cell list
        for i in range(len(c)):
            # get the state
            state = c[i]
            if g[tuple(state)] in ['.', 'S']:
                for a in range(len(directions)):
                    # get the state we reach after executing the action
                    new_state = state + directions[a]
                    # check if the next state exists in the cell list and save its index to j
                    j = np.where((c == new_state).all(axis=1))[0]
                    # update the probability transition matrix (use prob)
                    if j.size > 0:
                        assert j.size == 1
                        p[i, a, i] = 1. - prob
                        p[i, a, j] = prob
                    else:
                        p[i, a, i] = 1.  
        return p
    
    def compute_rewards(self, grid: np.ndarray, cell_list: list):
        """
        Compute the reward matrix.

        Args:
            grid (np.ndarray): 2D array representing the grid structure;
            cell_list (list): list of non-wall cells;

        Returns:
            The reward matrix;

        """
        g = np.array(grid)
        c = np.array(cell_list)
        n_states = len(cell_list)
        r = np.ones((n_states, 4, n_states)) * -0.01
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        def give_reward(poses:np.ndarray, rew:float):
            for x in poses:
                # check if the state x exists in the cell list and save its index to j
                j = np.where((c == x).all(axis=1))[0]
                if j.size > 0:
                    assert j.size == 1

                # iterate over the potential previous states
                for a in range(len(directions)):
                    prev_state = x - directions[a]
                    if prev_state.tolist() in c.tolist():
                        pass
                        # check if the state x exists in the cell list and save its index to i
                        i = np.where((c == prev_state).all(axis=1))[0]
                        if i.size > 0:
                            assert i.size == 1
                            j = j.copy() if j.size > 0 else i.copy()
                            # update the reward matrix
                            r[i[0], a, j[0]] = rew
                            
        pose_G = np.argwhere(g == 'G')
        give_reward(pose_G, 1.0)
        # give the lava -1.0 reward
        pose_lava = np.argwhere(g == '#')
        give_reward(pose_lava, -1.0)

        return r
    
    def initialize_state_distribution(self, cell_list: list, grid: np.ndarray):
        """
        Initialize the state distribution.

        Args:
            cell_list (list): list of non-wall cells;
            grid (np.ndarray): 2D array representing the grid structure;

        Returns:
            The initial state distribution;

        """
        g = np.array(grid)
        c = np.array(cell_list)
        n_states = len(c)
        mu = np.zeros(n_states)

        # get start position (cells of g with 'S')
        starts = np.argwhere(g == 'S')
        if len(starts) == 0:
            raise ValueError("CliffWalking grid must contain at least one 'S' (start) cell")
        # iterate over the starts, get their index in the cell list and update mu
        for start in starts:
            j = np.where((c == start).all(axis=1))[0]
            assert j.size == 1
        # (mu has to sum up to 1)
            mu[j] = 1. / len(starts)

        return mu
    
    def tf(self, i: int, j: int):
        H, W = self.grid.shape
        # transform the coordinates of the agent to the viewer coordinates
        x = j + 1
        y = H - i
        return np.array([x, y])
    
    def render(self):
        H, W = self.shape
        for row in range(0, H+1):
            for col in range(0, W+1):
                self.viewer.line(np.array([col, 0]), np.array([col, H+1]))
                self.viewer.line(np.array([0, row]), np.array([W+1, row]))
        # get the stariway status and the agent pose
        agent_x, agent_y = self.cell_list[self._state[0]]
        for i in range(0, H):
            for j in range(0, W):
                # fill the cell with a gray square
                self.viewer.square(self.tf(i,j), 0, 1, color='gray')
                # fill with red if lava
                if self.grid[i][j]=='#':
                    self.viewer.square(self.tf(i,j), 0, 1, color='red')
                # fill with green for the goal
                if self.grid[i][j]=='G':
                    self.viewer.square(self.tf(i,j), 0, 1, color='green')
                # fill with blue circle for the button
                # fill in with yellow arrow if agent
                if i==agent_x and j==agent_y:
                    self.viewer.arrow_head(self.tf(i,j), 1, 0, color='yellow')
        # render the viewer
        pygame.display.flip()
        # visualize
        pygame.image.save(self.viewer.screen,'src/reinforcement_learning/pictures/CliffWalking.png')
        img=cv2.imread('src/reinforcement_learning/pictures/CliffWalking.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        plt.pause(0.1)