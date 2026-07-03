from mushroom_rl.core import Core
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.environments import FiniteMDP
from mushroom_rl.policy import EpsGreedy
import numpy as np

from tqdm import trange
import scipy.stats as st
import matplotlib.pyplot as plt

from reinforcement_learning import CliffWalking, SARSA, QLearning


class Exp:
    """
    A class to launch experiments with Weights & Biases logging
    """
    def __init__(self, mdp: FiniteMDP, config_general: dict, exp_configs: list[dict]):
        # general config for all experiments
        self.config_general = config_general
        # initialize the MDP environment
        # eval() function in python is used to evaluate the string as a python expression
        #        we can pass the string of the class name and it will be evaluated as a class
        # **config_general['mdp_params'] is a dictionary unpacking
        #        the dictionary will be unpacked and passed as keyword arguments to the class
        self.mdp = mdp
        # the project is the name of the project for wandb
        self.project = config_general['project_name']
        # save the experiment configs
        self.exp_configs = exp_configs
        # number fo runs for each experiment
        self.n_runs = config_general['n_runs']
        self.n_epochs = config_general['n_epochs']
        # collect data for visualization
        self.data={}

    def run_experiments(self):
        fig = plt.figure()
        plt.suptitle(self.config_general['project_name'])
        legend_labels = []
        # iterate over the experiment configs and run the experiments
        for config in self.exp_configs:
            # we group the runs for the same agent by the group name
            # this allows us to compare the runs in the wandb dashboard
            self.group_name = config['group_name']
            # we load the config for the experiment
            self.config = config
            # Run the experiments
            Js=[]
            for i in range(self.n_runs):
              J = self.run_seed(i)
              Js.append(J)
            # save the results to a file using numpy
            np.save('src/reinforcement_learning/data/' + self.group_name, Js)
            # save the data for visualization
            self.data[self.group_name] = np.array(Js)
        # visualize the results
        self.visualize_results()

    def run_seed(self, i):
        # set the seed
        np.random.seed()
        # initialize the policy and the agent
        policy = eval(self.config['policy'])(**self.config['policy_params'])
        agent=eval(self.config['agent'])(self.mdp.info, policy, **self.config['agent_params'])
        run_name = self.group_name + '_' + str(i)
        # set the algorithm
        core = Core(agent, self.mdp)
        # learn over multiple epochs
        Js=np.zeros(self.n_epochs)
        for i in trange(self.n_epochs, desc = run_name):
          # run the algorithm
          core.learn(**self.config['learn_params'], quiet=True)
          # evaluate the agent
          dataset = core.evaluate(**self.config['eval_params'], quiet=True)
          # get J value
          J=np.mean(compute_J(dataset,self.mdp.info.gamma))
          Js[i]=J
        # save the agent
        agent.save('src/reinforcement_learning/models/agent_' + run_name + '.msh')
        return Js

    def visualize_results(self):
        # plt.figure(figsize=(10, 7))
        # iterate over data
        for k in self.data.keys():
            Js = self.data[k]
            # compute the mean and the 95% confidence interval of the results
            mean_Js = np.mean(Js, axis=0)
            lower_bound, upper_bound = st.t.interval(0.95, len(Js)-1, loc=mean_Js, scale=st.sem(Js))
            # plot the results
            plt.plot(mean_Js, label=k)
            plt.fill_between(np.arange(len(mean_Js)), lower_bound, upper_bound, alpha=0.3)
        plt.xlabel('epochs')
        plt.ylabel('J')
        plt.legend()
        # save the plot
        plt.savefig('src/reinforcement_learning/pictures/' + self.project + '.png')
        # plt.show()
        plt.close()


# define the experiment
config_general = {
    'project_name': 'TD_Learning',
    'n_runs': 5,
    'n_epochs':100,
    'mdp': 'CliffWalking',
    'mdp_params':{'gamma':0.9, 'horizon':25, 'prob':0.95},
}
exp_configs = [
    {
        'group_name': 'SARSA',
        'agent': 'SARSA',
        'agent_params': {'learning_rate':Parameter(0.1)},
        'policy': 'EpsGreedy',
        'policy_params': {'epsilon':Parameter(0.1)},
        'learn_params': {'n_episodes': 100, 'n_steps_per_fit': 1},
        'eval_params': {'n_episodes': 5}
    },
    {
        'group_name': 'QLearning',
        'agent': 'QLearning',
        'agent_params': {'learning_rate':Parameter(0.1)},
        'policy': 'EpsGreedy',
        'policy_params': {'epsilon':Parameter(0.1)},
        'learn_params': {'n_episodes': 100, 'n_steps_per_fit': 1},
        'eval_params': {'n_episodes': 5}
    },
]


# launch the experiment
grid_map=np.array(
           [['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['S', 'S', '#', '#', '#', '#', '#', '#', '#', '#', 'G']
            ]
        )
mdp=eval(config_general['mdp'])(grid_map, **config_general['mdp_params'])
launcher = Exp(mdp,config_general, exp_configs)
launcher.run_experiments()



def visualize_policy(Q, directions, grid_map, cell_list, label):
    """
    This function visualizes the policy matrices.
    Args:
        Q: action_value matrix
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    H,W=grid_map.shape
    pi = np.ones((H,W), dtype=int)*-1
    # fill in the policy according to the Q matrix values (iterate over the cell list)
    for i,c in enumerate(cell_list):
       pi[tuple(c)] = np.argmax(Q[i])
    plt.title("Policy matrix " + label)
    plt.imshow(pi, cmap='jet')
    # draw the policy arrows
    for i in range(H):
        for j in range(W):
          if pi[i,j]!=-1:
            plt.arrow(j, i, 0.25*directions[pi[i, j]][1], 0.25*directions[pi[i, j]][0], head_width=0.2, head_length=0.1, color='w')
    plt.show()



def load_and_visualize(agent_name, mdp: FiniteMDP, exp_configs):
    # get the group name from the agent name
    group_name = agent_name.split('_')[1]
    # get the config for the agent
    config = [config for config in exp_configs if config['group_name'] == group_name][0]
    # initialize the policy and the agent
    policy = eval(config['policy'])(**config['policy_params'])
    agent=eval(config['agent'])(mdp.info, policy, **config['agent_params'])
    # load the agent
    agent = agent.load('src/reinforcement_learning/models/' + agent_name)
    visualize_policy(agent.Q.table, mdp.directions, mdp.grid, mdp.cell_list, group_name)


for config in exp_configs:
  # visualize the first seed only
  load_and_visualize('agent_' + config['agent'] + '_0.msh', mdp, exp_configs)