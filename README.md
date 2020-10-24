# QLearning_MountainCar
Solving the Mountain Car environment of OpenAI using Q learning.
## Index
1) [Requirements](#requirements)
2) [Introduction](#introduction)
3) [Implemetation details](#implementation-details)
3) [Running The Code](#usage)
4) [Results](#results)
### Requirements 
The code works on python >= 3.0  
To install all the required libraries run :

```
! pip install -r requirements.txt
```

### Introduction
QLearning is an iterative method for ***MDP solving*** . Q Learning is a method for [model free control](http://web.stanford.edu/class/cs234/CS234Win2019/slides/lecture4_postclass.pdf). It is an [off policy](https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f) method. I have used the OpenAI gym environment for the algorithm.
ps://alex.smola.org/papers/2004/SmoSch04.pdf

### Implementation details
The self implementation has been done using numpy and openAI. **OpenAI** is a library used for testing and creating reinforcement learning algorithms. This code has been derived from the video lectures by [sentdex](https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7).

The hyperparameters involved here are :-
```
alpha = 0.1  // Learning rate
```
Another possibility of change is using the exploration value decay
```epsilon, epsilon_decay_value = 0.1 , 0.0005``

### Usage
To run the code , use
```
! python3 Mountain_Car_v0.py
```
The code uses discretization of the otherwise continuous states. For better results, Consider the following :-

1 Continuous State Linear Model <br>
2 Deep RL models <br>
3 comparison with SARSA, TD

### Results
Results can be found [here](Results/)
