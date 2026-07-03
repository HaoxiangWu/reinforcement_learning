import numpy as np
from mushroom_rl.environments import FiniteMDP

# Monte Carlo(TD(1)) Policy Evaluation
def evaluate_policy_mc(policy:np.ndarray, V:np.ndarray, returns_list:list, env:FiniteMDP, n_episodes=10):
    for _ in range(n_episodes):
        # Initialize state, done and episode list (to contain state and reward tuples)
        state = env.reset()
        done = False
        episode = []

        # Generate an episode following the policy
        while not done:
            action = policy[state] 
            # next_state, reward, done, _ = 
            next_state, reward, done, _ = env.step(action)
            # Store state and reward tuple in episode
            episode.append((state, reward))
            state = next_state
            if done:
                break

        G = 0
        for i, (state, reward) in enumerate(episode[::-1]):
            # Calculare return following this occurrence of state
            G = reward + env._mdp_info.gamma * G 
            # Add the return to the list of returns of the state
            returns_list[state[0]].append(G)
            # Update the value function as the mean of the returns of the state
            V[state[0]] = np.mean(returns_list[state[0]])

    return V


#TD(0) Policy Evaluation
def evaluate_policy_td(policy:np.ndarray, V:np.ndarray, env:FiniteMDP, alpha=0.1, n_episodes=10):
    for _ in range(n_episodes):
        # Initialize state and done
        state = env.reset()
        done = False

        # Generate an episode following the policy
        while not done:
            action = policy[state] 
            next_state, reward, done, _ = env.step(action) 
            # Compute the TD target as r + gamma * V(s')
            td_target = reward + env._mdp_info.gamma * V[next_state] if not done else reward 
            # Update the value function using the TD error
            V[state] = V[state] + alpha * (td_target - V[state]) 
            state = next_state
            if done:
                break
    return V


#TD(λ) Policy Evaluation
def evaluate_policy_td_lambda(policy:np.ndarray, V:np.ndarray, env:FiniteMDP, alpha=0.1, lmbda=0.5, n_episodes=10):
    n_states = env.p.shape[0]
    # eligibility_trace = np.zeros(n_states)

    for _ in range(n_episodes):
        # Initialize state, done and eligibility trace
        eligibility_trace = np.zeros(n_states) 
        state = env.reset()
        done = False

        # Generate an episode following the policy
        while not done:
            action = policy[state] 
            next_state, reward, done, _ = env.step(action) 
            # Compute the TD target as r + gamma * V(s')
            td_target = reward + env._mdp_info.gamma * V[next_state] if not done else reward 
            # Compute the TD error
            td_error = td_target - V[state] 
            # Update the eligibility trace
            eligibility_trace[state] += 1 

            for s in range(n_states):
                # Update the value function using the TD error and the eligibility trace
                V[[s]] = V[[s]] + alpha * td_error * eligibility_trace[s] 
                # Update the eligibility trace using the decay factor
                eligibility_trace[s] *= env._mdp_info.gamma * lmbda 

            if done:
                break

            state = next_state

    return V