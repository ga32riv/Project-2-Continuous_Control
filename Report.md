[//]: # (Image References)

[image1]: https://github.com/ga32riv/Project-2-Continuous_Control/blob/main/average%20reward%20episode.PNG "Average Reward"

[image2]: https://github.com/ga32riv/Project-2-Continuous_Control/blob/main/reward%20episode.PNG "Reward each Episode"

# Report

## Goal

The goal is to solve the environment option 2.
In this environment, a reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of agent is to maintain its position at the target location for as many time steps as possible.

## Steps
1. Examine the State and Action Spaces
2. Take Random Actions in the Environment (to learn how to use the Python API to control the agent and receive feedback from the environment)
3. Implement learning algorithm and check performance

## Algorithm
This project implements an policy-based method in order to handle the continious action space. A method called Deep Deterministic Policy Gradient (DDPG), described in the paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), is used in this project.
The agent was trained using [a single-agent DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a template

## NN Model Architecture

The *Actor* and the *Critic* Neural Networks have the same architecture:

- Fully connected layer - input (state size): 33 - output: 400 (ReLu activation function)

- Fully connected layer - input: 400 - output: 300 (ReLu activation function)

- Fully connected layer - input: 300 - output: 4 (action size, with tanh activation for the actor)

A [batch normalisation](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) is applied in the first fully connected layer

## Parameters

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit 
EPSILON_DECAY = 1e-6    # decay rate 

## Results
```
Episode 113 (222 sec)  -- 	Min: 36.8	Max: 39.5	Mean: 38.5	Mov. Avg: 30.0

Environment SOLVED in 113 episodes!	Moving Average =30.0 over last 100 episodes
```

![Average Reward][image1]
![Reward each Episode][image2]

## Ideas for Future Work
compare the results with other algorithm, for example try A2C and A3C

Perform a hyperparameter search
