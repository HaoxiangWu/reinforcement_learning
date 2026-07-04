from reinforcement_learning import SelfDefineMDP
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import cv2
import pygame
pygame.init()
screen = pygame.display.set_mode((500, 500))

class GridWorldEnv(SelfDefineMDP):
    '''
    There are five types of cells in GridWorld:

    'S' is the starting position where the agent is;
    'G' is the goal state;
    '.' is a normal cell;
    '*' is a hole, when the agent steps on a hole, it receives a negative reward and the episode ends;
    '#' is a wall, when the agent is supposed to step on a wall, it actually remains in its current state.
    The initial states distribution is uniform among all the initial states provided.

    We will go through an equivalent way to generate a grid world environment. Let's start by creating a sample grid:
    '''
    def _parse_cell_list(self):
        cell_list = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.grid[i, j] != '#':
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
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        p = np.zeros((n_states, 4, n_states))

        for i in range(len(c)):
            state = c[i]

            if g[tuple(state)] in ['.', 'S']:
                for a in range(len(directions)):
                    new_state = state + directions[a]
                    j = np.where((c == new_state).all(axis=1))[0]
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
            The reward matrix.

        """
        g = np.array(grid)
        c = np.array(cell_list)
        n_states = len(c)
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        r = np.zeros((n_states, 4, n_states))

        def give_reward(t, rew):
            for x in np.argwhere(g == t):
                j = np.where((c == x).all(axis=1))[0]

                for a in range(len(directions)):
                    prev_state = x - directions[a]
                    if prev_state.tolist() in c.tolist():
                        i = np.where((c == prev_state).all(axis=1))[0]
                        r[i, a, j] = rew

        give_reward('G', 1.)
        give_reward('*', -1.)
        give_reward('.', -.01)
        return r

    def initialize_state_distribution(self, cell_list: list, grid: np.ndarray):
        """
        Compute the initial states distribution.

        Args:
            grid (np.ndarray): 2D array representing the grid structure;
            cell_list (list): list of non-wall cells.

        Returns:
            The initial states distribution.

        """
        g = np.array(grid)
        c = np.array(cell_list)
        n_states = len(c)
        mu = np.zeros(n_states)

        for idx, (i, j)  in enumerate(cell_list):
            mu[idx] = 1. if g[i][j] == '.' else 0.

        if mu.sum() == 0:
            raise ValueError("GridWorldEnv grid must contain at least one '.' cell for initial state distribution")
        mu = mu / mu.sum()
        mu = mu.flatten()
        return mu
    
    def tf(self, i, j):
        # transform the coordinates of the agent to the viewer coordinates
        x = j + 1
        y = self.shape[0] - i
        return np.array([x, y])
    
    def render(self):
        for row in range(0, self.shape[0]+1):
            for col in range(0, self.shape[1]+1):
                self.viewer.line(np.array([col, 0]), np.array([col, self.shape[1]+1]))
                self.viewer.line(np.array([0, row]), np.array([self.shape[0]+1, row]))
        # get agent position
        agent_pose = (self._state//self.shape[1], self._state%self.shape[1])
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                # fill the cell with a gray square
                self.viewer.square(self.tf(i,j), 0, 1, color='gray')
                # fill with red if hole
                if self.grid[i][j]=='*':
                    self.viewer.square(self.tf(i,j), 0, 1, color='red')
                # fill the goal with green
                if self.grid[i][j]=='G':
                    self.viewer.square(self.tf(i,j), 0, 1, color='green')
                # fill with black for wall
                if self.grid[i][j]=='#':
                    self.viewer.square(self.tf(i,j), 0, 1, color='black')
                # fill in with yellow arrow if agent
                if i==agent_pose[0] and j==agent_pose[1]:
                    self.viewer.arrow_head(self.tf(i,j), 1, 0, color='yellow')
        # render the viewer
        pygame.display.flip()
        # visualize
        pygame.image.save(self.viewer.screen,'src/reinforcement_learning/pictures/gridWorldenv.png')
        img=cv2.imread('src/reinforcement_learning/pictures/gridWorldenv.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()