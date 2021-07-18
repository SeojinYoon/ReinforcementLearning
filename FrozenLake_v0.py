
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




