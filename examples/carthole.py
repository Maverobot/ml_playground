#!/usr/bin/env python3
import time
import gym
from gym import wrappers
import numpy as np
from math import *
import matplotlib.pyplot as plt

# Definition of this game: https://github.com/openai/gym/wiki/CartPole-v0

def inf_to_valid(value_list, min_value=-inf, max_value=inf):
    for i in range(len(value_list)):
        if value_list[i] > 1e30:
            value_list[i] = max_value
        elif value_list[i] < -1e30:
            value_list[i] = min_value
    return value_list

def observation_to_states(env, obs, n_buckets=5, min_value=-5, max_value=5):
    """ Map an observation to state """
    env_low = inf_to_valid(env.observation_space.low, min_value, max_value)
    env_high = inf_to_valid(env.observation_space.high, min_value, max_value)

    bucket_size = (env_high - env_low) / n_buckets
    s1 = floor((obs[0] - env_low[0]) / bucket_size[0])
    s2 = floor((obs[1] - env_low[1]) / bucket_size[1])
    s3 = floor((obs[2] - env_low[2]) / bucket_size[2])
    s4 = floor((obs[3] - env_low[3]) / bucket_size[3])
    return[s1, s2, s3, s4]

def select_action(env, actions_set):
    # use max_a to avoid value overflow
    max_a = np.max(actions_set)
    logits_exp = np.exp(actions_set - max_a)
    probs = logits_exp / np.sum(logits_exp)
    action = np.random.choice(env.action_space.n, p=probs)
    return action

def run_test_episode(env, policy=None, n_buckets=5, render=True, random_exploration_rate=0.1,
                     t_max=200, initial_lr=0.5, min_lr=0.1, gamma=1.0):
    observation = env.reset()
    total_reward = 0
    for t in range(t_max):
        learning_rate = max(min_lr, initial_lr * (0.85 ** (i//100)))
        if render:
            env.render()
        s1, s2, s3, s4 = observation_to_states(env, observation, n_buckets)
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy[s1,s2,s3,s4]
        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break
    return total_reward

def run_train_episode(env, policy=None, n_buckets=5, render=True, learning_rate=0.5, random_exploration_rate=0.1,
                      t_max=200, gamma=1.001):
    observation = env.reset()
    total_reward = 0
    for t in range(t_max):
        if render:
            env.render()
        s1, s2, s3, s4 = observation_to_states(env, observation, n_buckets)
        if policy is None or np.random.uniform(0, 1) < random_exploration_rate:
            action = env.action_space.sample()
        else:
            actions_set = policy[s1,s2,s3,s4]
            action = select_action(env, actions_set)

        observation, reward, done, info = env.step(action)
        total_reward += gamma ** t * reward

        #print("Pos: %f, Vel %f, Angle: %f, Vel_p: %f" % (state[0], state[1], state[2] * 180 / 3.1416, state[3]))
        # Update Q table
        s1_, s2_, s3_, s4_ = observation_to_states(env, observation, n_buckets)
        policy[s1, s2, s3, s4, action] +=  learning_rate * (reward + gamma * np.max(policy[s1_, s2_, s3_, s4_]) - policy[s1, s2, s3, s4, action])

        if done:
            break
    return policy, total_reward

if __name__ == '__main__':
    # Load CartPole-v0 game
    env = gym.make('CartPole-v0')

    # Set RNG seeds
    env.seed(0)
    np.random.seed(0)

    # Observation: [Cart Position, Cart Velocity, Pole Anlge, Pole Velocity At Top]
    # Action: 0 (Move to left) or 1 (Move to right)
    env.reset()

    # Q-learning initialization
    random_exploration_rate = 0.1
    n_buckets = 20
    n_iterations = 10000
    initial_lr=0.99
    min_lr=0.01
    q_table = np.zeros((n_buckets, n_buckets, n_buckets, n_buckets, env.action_space.n))

    # Run one episode
    rewards = []
    for i in range(n_iterations):
        learning_rate = max(min_lr, initial_lr * (0.85 ** (i//1000)))
        q_table, total_reward = run_train_episode(env, policy=q_table, n_buckets=n_buckets,
                                                  render=True, learning_rate=learning_rate)
        rewards.append(total_reward)

        # plot rewards
        plt.clf()
        plt.plot(rewards)
        plt.draw()
        plt.pause(0.0001)
        if i % 10 == 0:
            print('Iteration #%d -- Total reward = %d, learning rate = %f.' %(i+1, total_reward, learning_rate))

    solution_policy = np.argmax(q_table, axis=4)
    solution_policy_scores = [run_test_episode(env, solution_policy, n_buckets, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))

    env.close()

    input("Press Enter to continue...")
