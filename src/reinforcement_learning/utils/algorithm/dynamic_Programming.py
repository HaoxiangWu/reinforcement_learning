import numpy as np
from copy import deepcopy

def value_iteration(prob: np.ndarray, reward: np.ndarray, gamma: float, eps: float):
    """
    Value iteration algorithm to solve a dynamic programming problem.
    Args:
        prob (np.ndarray): transition probability matrix;
        reward (np.ndarray): reward matrix;
        gamma (float): discount factor;
        eps (float): accuracy threshold.
    Returns:
        The optimal value of each state.
    """
    # get the number of states and actions
    n_states = prob.shape[0]
    n_actions = prob.shape[1]
    # initialize the value matrix
    value = np.zeros(n_states)
    while True:
        # copy the old value
        value_old = deepcopy(value)
        # iterate over next state
        for state in range(n_states):
        # initialize the max value to a small number
        #   vmax = -np.inf
        # iterate over actions
            for action in range(n_actions):
                # get P^a_ss' and R_ss'^a
                prob_state_action = prob[state, action, :]
                reward_state_action = reward[state, action, :]
                # get the value estimate
                va = prob_state_action.T.dot(reward_state_action + gamma * value_old)
                # update the max value
                # vmax = max(va, vmax)
                value[state] = max(va, value[state])
            # update the value matrix with the maximum value
            #   value[state] = vmax
            # stop if the update is smaller that a threshold
        if np.linalg.norm(value - value_old) <= eps:
            break
    return value


def policy_iteration(prob:np.ndarray, reward:np.ndarray, gamma:float):
    """
    Policy iteration algorithm to solve a dynamic programming problem.
    Args:
        prob (np.ndarray): transition probability matrix;
        reward (np.ndarray): reward matrix;
        gamma (float): discount factor.
    Returns:
        The optimal value of each state and the optimal policy.
    """
    # get the number of states and actions
    n_states = prob.shape[0]
    n_actions = prob.shape[1]
    # initialize the policy and value matrices
    policy = np.zeros(n_states, dtype=int)
    value = np.zeros(n_states)
    # iterate until the policy and value stop changing
    changed = True
    while changed:
        # Policy Evaluation
        # I
        i = np.eye(n_states)
        # array to collect P^{pi} and R^{pi} over next states
        p_pi = np.zeros((n_states, n_states))
        r_pi = np.zeros(n_states)
        # iterate over next states
        for state in range(n_states):
            # get the action form the policy
            action = policy[state]
            # get P_s^{pi} and R_s^{pi}
            p_pi_s = prob[state, action, :]
            r_pi_s = reward[state, action, :]
            # update P^{pi} with P_s^{pi}
            p_pi[state, :] = p_pi_s
            # update R^{pi} with P_s^{pi}^T . R_s^{pi}
            r_pi[state] = p_pi_s @ r_pi_s
        # solve to get V : V (I-\gamma P^{pi}) = R^{pi}
        value = np.linalg.solve(i - gamma * p_pi, r_pi)
        # Policy iteration
        changed = False
        for state in range(n_states):
            # get the value of the state
            vmax = value[state]
            # iterate over all actions
            for action in range(n_actions):
                # if the action is different from the output of the policy thus the policy is not stable yet
                if action != policy[state]:
                    # get P_s,a and R_s,a
                    p_sa = prob[state, action]
                    r_sa = reward[state, action]
                    # update the value P_s,a^T . (R_s,a + \gamma * V)
                    va = p_sa @ (r_sa + gamma * value)
                    # if the value is bigger than the old value and not close (small difference)
                    # get the argmax action
                    if va > vmax and not np.isclose(va, vmax):
                        # update the policy
                        policy[state] = action
                        # update the max value
                        vmax = va
                        # update the changed flag
                        changed = True
    return value, policy


