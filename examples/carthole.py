#!/usr/bin/env python3

import gym
import time
from gym import wrappers

# Load CartPole-v0 scenari0
env = gym.make('CartPole-v0')
env.reset()

NUM_STEPS = 1000
for i in range(NUM_STEPS):
    done = env.step(env.action_space.sample())
    if done:
       env.reset()
    env.render()
