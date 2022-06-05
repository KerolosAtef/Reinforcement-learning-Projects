# Reinforcement Learning Project
## Table of Contents
- [Overview](#Overview)
- [Reuirements](#Requirements)
- [Training](#Training)
- [Evaluation](#Evaluation)

# Overview :
**Available Environments**<br>
	1-Taxi-v3<br>
	2-FrozenLake-v1<br>
	3-CliffWalking-v0<br>
you can choose from them and the defualt one is Taxi
**Proplem definition :**
There are 4 locations (labeled by different letters), and our job is to pick up
the passenger at one location and drop him off at another. We receive +20
points for a successful drop-off and lose 1 point for every time-step it
takes. There is also a 10 points penalty for illegal pick-up and drop-off
actions.
**Introduction**
In this project we will implement the Q-leaning algorithm and will see how the decay of the hyperparameter such as learning rate and discount factor and eplison will effect the results and we will implement a grid search to select the best parameters.

# Requirements :
It is required to setup this libraries to run the project
```python
!pip install gym
!pip install numpy
```
# Defualt environment info
[![Taxi-env](https://storage.googleapis.com/lds-media/images/Reinforcement_Learning_Taxi_Env.width-1200.png "Taxi-env")](https://storage.googleapis.com/lds-media/images/Reinforcement_Learning_Taxi_Env.width-1200.png "Taxi-env")
The job is to pick up the passenger at one location and drop them off in another. Here are a few things that we'd love our taxi to take care of:
- Drop off the passenger to the right location.
- Save passenger's time by taking minimum time possible to drop off
- Take care of passenger's safety and traffic rules

# Training :
## Random Action :
**Trying random actions to see how the agent movements**<br><br>
![Random Actions](https://drive.google.com/uc?export=view&id=17n6oBPFjce-AjRYUR9NDVb39VYSB3LP4)
## Q-Learning 
**After the agent has been trained**<br><br>
![Q-learning](https://drive.google.com/uc?export=view&id=1UDjqQLfPtllelNdZkTptLZKHuXoG3AOD)<br><br>
We can notice the difference and how the agent has been trained
## Decaying hyper parameters while training

![Decay](https://drive.google.com/uc?export=view&id=1U6W79ftVPSUZ_Wcv762UP_p3dXmNc4fr)
# Evaluation 
![Eval_100](https://drive.google.com/uc?export=view&id=1BAmzOSUVoL148BMOrDsRpKCP5jkn0RCN)
# Grid search 
**We used a brute force algorithm to get the best hyper parameter**<br><br>
![Grid search values](https://drive.google.com/uc?export=view&id=1rS39uEeHeYdOn5SXAYKhAkGykmt9dkSX)

**Experiments table**<br><br>
![parameters](https://drive.google.com/uc?export=view&id=1vXLt7JjRLveEa8oHBtU6jzd9GUvSiPft)<br><br>
**Best hyperparameters :**
```python
alpha=0.9 ,gamma=0.9, epsilon=0.9
```

**Grid search evaluation**<br><br>
![Grid search](https://drive.google.com/uc?export=view&id=1KbCs0L-71cGMLk5-6TaMnYl1QbJtH5fp)<br><br>
**Plot the Penalties and number of epochs in each iteration**<br><br>
![errors](https://drive.google.com/uc?export=view&id=10cZzQKTRVIIF0WiS1Pj7rvlTbyFAytoy)<br><br>
