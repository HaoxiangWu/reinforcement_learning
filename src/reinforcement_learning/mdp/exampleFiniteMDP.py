import numpy as np

from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.environments import FiniteMDP



class ExampleGridWorldEnv(FiniteMDP):
    """
    Example of environment class construction and usage of Mushroom FiniteMDP class
    It is a mini grid environment with bramble bushes that hurt the actor if crossed.
    The actor is spawn in the bottom-left quadrant of the grid and the goal is in the top-right corner.
    There is a pair of shears in top-left corner that allows the actor to prune the bushes.
    The agent can move in the four directions.
    The episode ends when the agent reaches the goal.
    """
    def __init__(self, size=(10, 10), n_obs=10, gamma=0.99, horizon=100, prob=1):
        # initialize the base class

        # set the size of the grid
        self._size = size

        ###
        # Obstacles positions
        self.n_obs = n_obs
        self.obstacle_x_idx_B = np.random.randint(1, self._size[0]-1, n_obs)
        self.obstacle_y_idx_B = np.random.randint(1, self._size[1]-1, n_obs)
        ###

        # set the probability of success for each action
        self.prob = prob

        # change in coordinates for each action
        self.directions = [[0, 1], [-1, 0], [0, -1], [1, 0]]

        # define a viewer for visualization
        self._viewer = Viewer(self._size[0]+1, self._size[1]+1, 500, 500)

        # record last action to use it in the visualization
        self._last_action = [0]

        # build two finite MDPs for the 2 states of the bushes
        # initial state distribution
        mu = self.initialize_state_distribution()
        # transition matrix
        p = self.compute_probabilities(self.prob)
        # reward matrix
        r = self.compute_rewards()

        # call the super class
        super().__init__(p, r, mu, gamma, horizon)

        # reset the environment
        self.reset()

    def initialize_state_distribution(self):
        # Create the initial state distribution.

        # Initialize the state distribution
        mu = np.zeros((2,self._size[0],self._size[1]))

        mu[0,self._size[0]//2:, :self._size[1]//2] = 1 # fill the bottom left quarter with 1 for s=0

        # normalize the distribution
        mu = mu/np.sum(mu)

        # flatten the distribution
        mu = mu.flatten()

        return mu

    def state_(self, s, i_a, j_a):
        # Get state index from agent position.

        idx = s*self._size[0]*self._size[1]+i_a*self._size[1]+j_a

        # Notice that:
        #     s = idx // (self._size[0]*self._size[1])
        #     i_a = (idx % (self._size[0]*self._size[1])[0] // self._size[1]
        #     j_a = (idx % (self._size[0]*self._size[1])[0] % self._size[1]

        return idx

    def compute_probabilities(self, prob):
        # Compute the transition probabilities.

        # Initialize transition probability matrix with 0
        # The size = (observation space shape) x (action space shape) x (observation space shape)
        p = np.zeros((2*self._size[0]*self._size[1], 4, 2*self._size[0]*self._size[1]))

        # iterate over the states and fill the transition probability matrix
        for s in range(2):
            for i_a in range(self._size[0]):
                for j_a in range(self._size[1]):

                    # to set the goal as a terminal state
                    if i_a==0 and j_a==self._size[1]-1:
                        continue

                    # skip if you collect the shears
                    if s==0 and i_a==0 and j_a==0:
                        p[self.state_(0,0,0), : , self.state_(1,0,0)] = 1
                        continue

                    # iterate over the actions
                    for a in range(4):
                        # compute the new position of the actor
                        new_i_a = i_a + self.directions[a][0]
                        new_j_a = j_a + self.directions[a][1]

                        # if the new position is a wall, stay in the same position
                        if new_i_a < 0 or new_i_a >= self._size[0] or new_j_a < 0 or new_j_a >= self._size[1]:
                            new_i_a = i_a
                            new_j_a = j_a

                        p[self.state_(s,i_a,j_a), a, self.state_(s,new_i_a,new_j_a)] += prob    # probability to perform the action
                        p[self.state_(s,i_a,j_a), a, self.state_(s,i_a,j_a)] += 1-prob          # probability to fail to perform the action

        return p

    def compute_rewards(self):
        # Compute the rewards.

        # reward matrix - initialized with -0.01 reward for each transition to encourage the agent to reach the goal as fast as possible
        # the size is (observation space shape) x (action space shape) x (observation space shape)
        r = np.ones((2*self._size[0]*self._size[1], 4, 2*self._size[0]*self._size[1])) *-0.01

        # iterate over the states and fill the reward matrix
        for s in range(2):
            for i_a in range(self._size[0]):
                for j_a in range(self._size[1]):

                    # iterate over the actions
                    for a in range(4):
                        # compute the new position
                        new_i_a = i_a + self.directions[a][0]
                        new_j_a = j_a + self.directions[a][1]

                        # skip if the new position is a wall
                        if new_i_a<0 or new_i_a>=self._size[0] or new_j_a<0 or new_j_a>=self._size[1]:
                            continue

                        # if you collect the shear, the reward is 0.1
                        if new_i_a==0 and new_j_a==0:
                            r[self.state_(0,i_a,j_a), a, self.state_(0,new_i_a,new_j_a)] = 0.1
                            r[self.state_(1,i_a,j_a), a, self.state_(1,new_i_a,new_j_a)] = -0.01

                        # if the new position is a bush and the agent is not equipped with the shears, the reward is -0.1
                        if s == 0:
                            for obs in range(self.n_obs):
                                if new_i_a == self.obstacle_x_idx_B[obs] and new_j_a == self.obstacle_y_idx_B[obs]:
                                    r[self.state_(s,i_a,j_a), a, self.state_(s,new_i_a,new_j_a)] = -0.1
                                    r[self.state_(s,i_a,j_a), a, self.state_(s,new_i_a,new_j_a)] = -0.1

                        # if the new position is the goal, the reward is 1
                        if new_i_a == 0 and new_j_a == self._size[1]-1:
                            r[self.state_(s,i_a,j_a), a, self.state_(s,new_i_a,new_j_a)] = 1.0

        return r

    def tf(self, i, j):
        # Transform the coordinates of the agent to the viewer coordinates

        x = j + 1
        y = self._size[0] - i

        return np.array([x, y])

    def render(self):
        # Render the environment

        for row in range(0, self._size[0]+1):
            for col in range(0, self._size[1]+1):
                self._viewer.line(np.array([col, 0]), np.array([col, self._size[1]+1]))
                self._viewer.line(np.array([0, row]), np.array([self._size[0]+1, row]))

        for i in range(0, self._size[0]):
            for j in range(0, self._size[1]):
                self._viewer.square(self.tf(i,j), 0, 1, color='limegreen')        # fill the cell with a limegreen square

        self._viewer.square(self.tf(0, self._size[1]-1), 0, 1, color='orangered') # draw a red square for the goal in the top-right corner (goal position)

        # get the coordinates of the agent and the shear from the state
        state = self._state%self.state_(1,0,0)
        s = self._state//(self._size[0]*self._size[1])
        agent_pose = (state[0]//self._size[1], state[0]%self._size[1])

        for i in range(0, self._size[0]):
            for j in range(0, self._size[1]):
                for obs in range(self.n_obs):
                    if s == 0:
                        self._viewer.circle(self.tf(0,0), 0.5, color='yellow')
                        if i == self.obstacle_x_idx_B[obs] and j == self.obstacle_y_idx_B[obs]:
                            self._viewer.square(self.tf(i,j), 0, 1, color='forestgreen')
                    else:
                        if i == self.obstacle_x_idx_B[obs] and j == self.obstacle_y_idx_B[obs]:
                            self._viewer.square(self.tf(i,j), 0, 1, color='darkgoldenrod')


        # get the rotation of the agent from the last action
        agent_rotation = self._last_action[0]

        self._viewer.arrow_head(self.tf(agent_pose[0], agent_pose[1]), 1, agent_rotation * np.pi/2 , color='deepskyblue') # draw the agent as a blue arrow head

        # render the viewer
        self._viewer.display(0.1)