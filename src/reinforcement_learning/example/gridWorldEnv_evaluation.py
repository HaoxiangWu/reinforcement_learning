from reinforcement_learning import GridWorldEnv, policy_iteration, value_iteration
import matplotlib.pyplot as plt
from reinforcement_learning import evaluate_policy_td, evaluate_policy_td_lambda, evaluate_policy_mc


# Define the grid structure
import numpy as np

grid=np.array(
    [['.', '.', '.', '.', '.'],
     ['.', '.', '.', '.', '.'],
     ['#', '#', '#', '#', '.'],
     ['.', '.', '.', '.', '.'],
     ['G', '*', '*', '.', '.']
     ]
)
env = GridWorldEnv(grid)
directions = env.directions
# env.render()


def visualize_value_matrices(V: np.ndarray, H: int, W: int, cell_list: list, label: str):
    """
    This function visualizes the value matrices.
    Args:
        V (np.ndarray): value matrix with a size of [n_states]
        H (int): height of the grid
        W (int): width of the grid
        cell_list (list): the MDP cell list
        label (str): the title of the figure
    """
    # reshape the value matrices
    V_ = np.zeros((H,W))
    #plt.imshow(V, cmap='jet')
    plt.title(label)
    # write the values in the cells
    for idx, (i, j) in enumerate(cell_list):
        V_[i][j] = V[idx]
        plt.text(j, i, round(V[idx], 2), ha="center", va="center", color="w")
    plt.imshow(V_, cmap='jet')
    plt.show()


def visualize_policy(pi: np.ndarray, H: int, W: int, cell_list: list, label: str):
    """
    This function visualizes the policy matrices.
    Args:
        pi (np.ndarray): policy matrix with a size of [n_states]
        H (int): height of the grid
        W (int): width of the grid
        cell_list (list): the MDP cell list
        label (str): the title of the figure
    """
    # reshape the policy matrices
    pi_ = np.zeros((H,W))
    plt.title(label)
    # draw the policy arrows
    for idx, (i, j) in enumerate(cell_list):
        # x,y are reversed from the indics (i,j) -> (j,i)
        pi_[i][j] = pi[idx]
        plt.arrow(j, i, 0.25*directions[pi[idx]][1], 0.25*directions[pi[idx]][0], head_width=0.2, head_length=0.1, color='w')
    plt.imshow(pi_, cmap='jet')
    plt.show()


V = value_iteration(env.p, env.r, env._mdp_info.gamma, eps=1e-6)
# visualize_value_matrices(V, env.shape[0], env.shape[1], env.cell_list, "Value Iteration")


# visualize the policy iteration
V, pi = policy_iteration(env.p, env.r, gamma=0.99)
# visualize_value_matrices(V, env.shape[0], env.shape[1], env.cell_list, "Policy_Iteration_Value")
# visualize_policy(pi, env.shape[0], env.shape[1], env.cell_list,"Policy_Iteration_Policy")


# Initialize the value function
V_MC_init = np.zeros(env.p.shape[0])
V_TD_init = np.copy(V_MC_init)
V_TD_lambda_init = np.copy(V_MC_init)


# Evaluate the policy using Monte Carlo
# Initialize the list of returns(s) for all states
returns_list = [[] for _ in range(env.p.shape[0])]
np.random.seed(0)
V_MC = evaluate_policy_mc(pi, V_MC_init, returns_list, env, n_episodes=1000)
# print(V_MC.shape)

# Evaluate the policy using TD
np.random.seed(0)
V_TD = evaluate_policy_td(pi, V_TD_init, env, n_episodes=1000)
# print(V_TD.shape)

# Evaluate the policy using TD(\lambda)
np.random.seed(0)
V_TD_lambda = evaluate_policy_td_lambda(pi, V_TD_lambda_init, env, lmbda=0.5, n_episodes=1000)
# print(V_TD_lambda.shape)
print('Sanity check: TD=TD(lambda=0.5)', V_TD==V_TD_lambda)


visualize_value_matrices(V_MC, grid.shape[0], grid.shape[1], env.cell_list, 'MC')

visualize_value_matrices(V_TD, grid.shape[0], grid.shape[1], env.cell_list, 'TD(0)')

visualize_value_matrices(V_TD_lambda, grid.shape[0], grid.shape[1], env.cell_list, 'TD(lambda=0.5)')