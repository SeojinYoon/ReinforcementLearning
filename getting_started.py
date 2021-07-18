
# https://gym.openai.com/docs/

"""
Minimum example of environment
"""
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

"""
Getting information from agent
"""
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset() # It returns initial observation
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action) # step returns that observation, reward, done flag, info
        # info means that diagnostic information useful for debugging.

        # done means that the game is over
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

"""
Spaces
"""
import gym
env = gym.make('CartPole-v0')
print(env.action_space) # Action space
#> Discrete(2)
print(env.observation_space) # observation space
#> Box(4,)
print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])

from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8

