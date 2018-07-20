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

# NN parameters
input_n = 4
output_n = 2

# Description of TF graph
state = tf.placeholder(shape=[1,input_n], dtype=tf.float32)
hidden = tf.layers.dense(state, 128, activation=tf.nn.relu)
hidden = tf.layers.dropout(hidden, 0.1)
hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu)
hidden = tf.layers.dropout(hidden, 0.1)
Q_values = tf.layers.dense(hidden, 2, activation=tf.nn.relu)
action = tf.argmax(Q_values, 1)

next_Q_values = tf.placeholder(shape=[1,output_n], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_Q_values- Q_values))
opt = tf.train.AdamOptimizer(learning_rate=0.1)
opt_op = opt.minimize(loss)

# Learning parameters
gamma = 0.99
random_exploration_rate = 0.5

# Training the network
init = tf.global_variables_initializer()
num_episodes = 3000

# Create lists to contain total rewards
rewards = []
actions = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        observation = env.reset()
        observation = np.reshape(observation, (1, 4))
        total_rewards = 0
        done = False
        #The Q-Netwo]rk
        for t in range(200):
            # Choose an action by greedily (with e chance of random action) from the Q-network

            a, Q = sess.run([action, Q_values],feed_dict={state:observation})

            # Explore randomly
            if np.random.uniform(0, 1) < random_exploration_rate:
                a[0] = env.action_space.sample()

            actions.append(a[0])

            # Run one step simulation and get feedback
            observation, reward, done, _ = env.step(a[0])
            # env.render()
            observation = np.reshape(observation, (1, 4))

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Q_values,feed_dict={state:observation})

            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = Q
            targetQ[0, a[0]] = reward + gamma * maxQ1

            # Train our network using target and predicted Q values
            _ = sess.run([opt_op],feed_dict={state:observation,next_Q_values:targetQ})
            total_rewards += reward
            s = observation
            if done:
                break
        rewards.append(total_rewards)
        if i % 10 == 0:
            # plot rewards
            plt.figure(1)
            plt.clf()
            plt.plot(rewards)
            plt.figure(2)
            plt.clf()
            plt.hist(actions)
            plt.draw()
            plt.pause(0.01)
            print('Iteration #%d -- Total reward = %d.' %
                  (i, total_rewards))

env.close()
input("Press Enter to continue...")

