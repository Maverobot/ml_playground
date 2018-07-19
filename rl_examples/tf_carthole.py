#!/usr/bin/env python3
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# Definition of this game: https://github.com/openai/gym/wiki/CartPole-v0
# This code is inspired by https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

env = gym.make('CartPole-v0')
tf.reset_default_graph()

# Description of TF graph
state = tf.placeholder(shape=[4,1], dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([12, 4], 0, 0.01), name="W1")
b1 = tf.Variable(tf.random_normal([12, 1], 0, 0.01), name="b1")
W2 = tf.Variable(tf.random_uniform([2, 12], 0, 0.01), name="W2")
b2 = tf.Variable(tf.random_normal([2, 1], 0, 0.01), name="b2")
H = tf.nn.tanh(tf.add(tf.matmul(W1, state), b1))
Q_values = tf.nn.tanh(tf.add(tf.matmul(W2, H), b2))
action = tf.argmax(Q_values, 0)
next_Q_values = tf.placeholder(shape=[2,1], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_Q_values- Q_values))
opt = tf.train.AdamOptimizer(learning_rate=0.1)
opt_op = opt.minimize(loss)

# Learning parameters
gamma = 0.99
random_exploration_rate = 0.1

# Training the network
init = tf.global_variables_initializer()
num_episodes = 100000

# Create lists to contain total rewards
rewards = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        observation = env.reset()
        observation = np.reshape(observation, (4, 1))
        total_rewards = 0
        done = False
        #The Q-Netwo]rk
        for t in range(200):
            # Choose an action by greedily (with e chance of random action) from the Q-network

            a, Q = sess.run([action, Q_values],feed_dict={state:observation})

            # Get the action as scalar
            a = a[0]

            # Explore randomly
            if np.random.uniform(0, 1) < random_exploration_rate:
                a = env.action_space.sample()

            # Run one step simulation and get feedback
            observation, reward, done, _ = env.step(a)
            env.render()
            observation = np.reshape(observation, (4, 1))

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Q_values,feed_dict={state:observation})

            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = Q
            targetQ[a] = reward + gamma * maxQ1

            # Train our network using target and predicted Q values
            _ = sess.run([opt_op],feed_dict={state:observation,next_Q_values:targetQ})
            total_rewards += reward
            s = observation
            if done:
                break

        # plot rewards
        plt.clf()
        plt.plot(rewards)
        plt.draw()
        plt.pause(0.0001)
        if i % 10 == 0:
            print('Iteration #%d -- Total reward = %d.' %
                  (i, total_rewards))
        rewards.append(total_rewards)

env.close()
input("Press Enter to continue...")

