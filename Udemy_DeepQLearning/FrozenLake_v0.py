
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/learn/lecture/17009504#overview
"""
State

0 1 2 3
4 5 6 7
8 9 10 11
12 13 14 15
"""

"""
Action
0: Left
1: Down
2: Right
3: Up
"""
import gym
import numpy as np
import matplotlib.pylab as plt

env = gym.make("FrozenLake-v0")

n_games = 1000
win_pct = []
scores = []
for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
plt.plot(win_pct)
plt.show()

# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/learn/lecture/17009508#notes
import gym
import numpy as np
import matplotlib.pylab as plt

env = gym.make("FrozenLake-v0")

n_games = 1000
win_pct = []
scores = []

game_sample_cnt = 10

# key, state : value, policy
policy = {0: 1,
          1: 2,
          2: 1,
          3: 0,
          4: 1,
          6: 1,
          8: 2,
          9: 1,
          10: 1,
          13: 2,
          14: 2}

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0

    while not done:
        action = policy[obs] # take action from policy
        obs, reward, done, info = env.step(action)
        score += reward

    scores.append(score)

    if i % game_sample_cnt == 0:
        average = np.mean(scores[-game_sample_cnt:]) # 10개의 데이터만 mean
        win_pct.append(average)
plt.plot(win_pct)
plt.show()

"""
https://www.udemy.com/course/deep-q-learning-from-paper-to-code/learn/lecture/17009516#content

Temporal Difference Learning
"""
import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}

        self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        """
        :param state: current state
        :return: action
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        """
        Update Q value in accordance with state transition

        :param state: current state
        :param action: action
        :param reward: state transition's reward
        :param state_: next state
        """
        actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions) # 어떤 State에서의 모든 Action중 Q-value가 가장 큰 Action

        # Refinement
        # New estimate = old estimate + step size( target - old estimate )
        # Q(s_t, a_t) = Q(s_t, a_t) + α( R[t+1] + γ*maxQ(s[t+1], a[max]) - Q(s[t], a[t]) )
        self.Q[(state, action)] += self.lr*(reward + self.gamma * self.Q[(state_, a_max)] - self.Q[(state, action)])

        self.decrement_epsilon()

import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v0")
agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.9999995, n_actions=4, n_states=16)

scores = []
win_pct_list = []
n_games = 500000

for i in range(n_games):
    done = False
    observation = env.reset()
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.learn(observation, action, reward, observation_)
        score += reward
        observation = observation_
    scores.append(score)
    if i % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pct_list.append(win_pct)
        if i % 1000 == 0:
            print("episode ", i, "win pct %.2f" % win_pct,
                  "epsilon %.2f" % agent.epsilon)

plt.plot(win_pct_list)
plt.show()

