<h1 align="center">CardPole<h1>
  
## Q-Learning in Reinforcement learning with NUMPY

In this game, you can push the bottom cart left and right to keep the pole from falling. If the pole is tilted at a certain angle, it will fall down, which means the game is over.

<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_0_resule.gif" width="50%"></p>
  
After each action, four environmental values such as the cart position, cart speed, pole angle, and pole speed of the action are returned.

After each action, if the pole does not fall down, you can get one point, which can be continuously accumulated. If the total score is 195 points, it means that you have pushed 195 times without falling down.

## Q-Learning

The Q of Q-Learning is the value function Q(s, a), which can represent the value obtained when taking action (a) in state (s).
The first example - cartpole1.py, will create a Q Table, first understand the formula.

<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_QLearning_resule.png" width="50%"></p>
  
Then according to the execution results of cartpole1.py, the average result of 5000 times falls between 10 and 15, with occasional increases, but not very frequently.
  
<p align="center">▼ cartpole1.py</p>
  
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_1_resule.png" width="50%"></p>

## Exploration-exploitation dilemma
  
The first example can be said that AI is looking for the highest value options at present, so if the value of the data opened is very low, it will be difficult to increase the subsequent exploration.

The second example is the interaction between exploration and exploitation.
Choose to take exploration or exploitation with a specific probability, also known as theεgreedy strategy.
  
<p align="center">▼ cartpole2.py</p>
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_2_resule.png" width="50%"></p>
It can be seen that the peak value of the second result is more than first time.
  
## Increase the frequency of exploration at the beginning of learning

At the beginning of the learning, the uncertainty is relatively high, so the frequency of exploration can be increased, and the program can explore new knowledge; in the later stage of learning, the unexplored knowledge is reduced, and the program learning is more comprehensive. At this time, the frequency should be reduced to use the learning process obtained experience.

In the third example, get_action() adds an **episode parameter** to let the program determine whether the current stage is the early or late stage of learning (theεvalue is higher in the early stage, and theεvalue in the later stage is lower).

<p align="center">▼ cartpole3.py</p>
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_3_resule.png" width="50%"></p>
  
  
## Add punishment
  
In the fourth example, a penalty system is added, and the reward points after failure are changed to negative values, that is, points are deducted when the pole falls.

<p align="center">▼ cartpole4.py</p>
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_4_resule.png" width="50%"></p>
  
  
## Polocy Gradient

Finally try the final example with Policy Gradient(策略梯度法).
The previous examples are based on the state to select the most valuable action, but Policy Gradient is to output the probability of different actions according to the state, and then choose which action to take according to the probability value.

This example uses a single-layer neural network with no intermediate layers and no activation function.
Polocy Gradient is a neural network learning method with the goal of maximizing the reward function. There are two rules:
  
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_6.png" width="50%"></p>
  
Therefore, when all the steps in the round are less than 200, the more steps, the higher the reward value. The program will play 10 rounds at a time, collect the rewards of these 10 rounds and then update the neural network parameters, the goal is to maximize this parameter.
  
## Update W parameter

Like the neural network introduced earlier, partial differential will be used to update each W parameter, but the previous one is to minimize the loss function, so the partial differential value is subtracted to update W. This example is to let the reward function Maximize, so add the partial differential value to update the W parameter
  
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_eta_function.png" width="15%"></p>
<p align="center">η is learning rate (a.k.a. eta)</p>

<p align="center">▼ cartpole5.py</p>
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/CartPole/cartPole_5_resule.png" width="50%"></p>
  
The final example performed very well, surpassing 195 in about 350 rounds.
