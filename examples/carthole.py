#!/usr/bin/env python3
import time
import gym
from gym import wrappers
import numpy as np
from math import *

# Definition of this game: https://github.com/openai/gym/wiki/CartPole-v0

def observation_to_states(env, obs, n_buckets=5):
    """ Map an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    bucket_size = (env_high - env_low) / n_buckets
    s1 = floor((obs[0] - env_low[0]) / bucket_size[0])
    s2 = floor((obs[1] - env_low[1]) / bucket_size[1])
    s3 = floor((obs[2] - env_low[2]) / bucket_size[2])
    s4 = floor((obs[3] - env_low[3]) / bucket_size[3])
    return[s1, s2, s3, s4]

def select_action(env, actions_set):
    logits_exp = np.exp(actions_set)
    probs = logits_exp / np.sum(logits_exp)
    action = np.random.choice(env.action_space.n, p=probs)
    return action

def run_episode(env, policy=None, n_buckets=5, t_max=100, learning_rate=0.5, gamma=1.0, render=True):
    observation = env.reset()
    total_reward = 0
    for t in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            s1, s2, s3, s4 = observation_to_states(env, observation, n_buckets)
            actions_set = policy[s1,s2,s3,s4]
            action = select_action(env, actions_set)
        observation, reward, done, info = env.step(action)
        total_reward += reward

        #print("Pos: %f, Vel %f, Angle: %f, Vel_p: %f" % (state[0], state[1], state[2] * 180 / 3.1416, state[3]))
        # Update Q table
        s1_, s2_, s3_, s4_ = observation_to_states(env, observation, n_buckets)
        policy[s1, s2, s3, s4, action] += learning_rate * (reward + gamma * np.max(policy[s1_, s2_, s3_, s4_] - policy[s1, s2, s3, s4, action]))
        if done:
            break
    return policy, total_reward

if __name__ == '__main__':
    # Load CartPole-v0 game
    env = gym.make('CartPole-v0')

    # Observation: [Cart Position, Cart Velocity, Pole Anlge, Pole Velocity At Top]
    # Action: 0 (Move to left) or 1 (Move to right)
    env.reset()

    # Q-learning initialization
    n_buckets = 5
    n_iterations = 100
    q_table = np.zeros([n_buckets, n_buckets, n_buckets, n_buckets, env.action_space.n])

    # Run one episode
    for i in range(n_iterations):
        q_table_old = q_table
        q_table, total_reward = run_episode(env, policy=q_table, n_buckets=n_buckets)
        print(np.sum(q_table - q_table_old))
        if i % 10 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))

    env.close()

