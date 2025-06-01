# EX_03 - POLICY ITERATION ALGORITHM

## AIM

To implement the Policy Iteration algorithm using Python in the FrozenLake-v1 environment and evaluate the resulting optimal policy based on its success rate and mean return.

## PROBLEM STATEMENT

The FrozenLake environment is a benchmark reinforcement learning task where an agent must navigate a grid-based frozen surface to reach a goal state without falling into holes. The environment is stochastic, meaning that the agent's actions have uncertain outcomes due to slippery tiles.

The environment is modeled as a Markov Decision Process (MDP) with:

- States: Each cell in the grid.

- Actions: Move left, right, up, or down.

- Transition Probabilities: Due to the slippery nature, intended actions might not always be executed.

- Rewards: +1 for reaching the goal (G), 0 otherwise.

The goal is to determine the optimal policy—a mapping from states to actions—that maximizes the cumulative reward. This is done using the Policy Iteration algorithm, which iteratively evaluates and improves a policy until convergence.

## POLICY ITERATION ALGORITHM:

Policy Iteration is a classic method for solving MDPs and consists of two main steps: Policy Evaluation and Policy Improvement, repeated until the policy stabilizes.

### Steps of the Policy Iteration Algorithm:

![image](https://github.com/user-attachments/assets/9f731f3b-cfac-4400-989f-2a3a585465ad)


### Policy Improvement Function:

```
def policy_improvement(V, P, gamma=1.0):
    def improved_policy(s):
        action_values = []
        for a in range(len(P[s])):
            q_sa = 0
            for prob, next_state, reward, done in P[s][a]:
                q_sa += prob * (reward + gamma * V[next_state] * (not done))
            action_values.append(q_sa)
        return np.argmax(action_values)
    return improved_policy

```

### Policy Iteration Function

```
def policy_iteration(P, gamma=1.0, theta=1e-10):
    pi = lambda s: 2  # start with all RIGHT actions
    stable = False
    while not stable:
        V = policy_evaluation(pi, P, gamma, theta)
        new_pi = policy_improvement(V, P, gamma)
        stable = True
        for s in range(len(P)):
            if pi(s) != new_pi(s):
                stable = False
                break
        pi = new_pi
    return V, pi
     
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy


![image](https://github.com/user-attachments/assets/14683505-0333-45c8-a14a-5193dc938a07)



![image](https://github.com/user-attachments/assets/13e268c7-0030-4695-9f54-d21985141e69)



### 2. Policy, Value function and success rate for the Improved Policy


![image](https://github.com/user-attachments/assets/38b43c0a-6d66-48c9-9258-9b148563a654)


![image](https://github.com/user-attachments/assets/32eb3558-b78a-4ad5-b24c-794d5cd49ce8)


![image](https://github.com/user-attachments/assets/37812cb1-2747-4ba3-8f84-d8f9d8808162)



### 3. Policy, Value function and success rate after policy iteration


![image](https://github.com/user-attachments/assets/8e65056d-182c-44d3-b69d-4f35809c5a1c)


![image](https://github.com/user-attachments/assets/62a27706-8e00-4656-a485-2e2dc615923c)

![image](https://github.com/user-attachments/assets/3975e622-96d0-4f79-bbeb-0b01d90b2438)


## RESULT:

Thus to implement the Policy Iteration algorithm using Python in the FrozenLake-v1 environment and evaluate the resulting optimal policy based on its success rate and mean return is successfully implemented.
