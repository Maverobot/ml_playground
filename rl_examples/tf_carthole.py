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
mapping = tf.Variable(tf.random_uniform([2, 4], 0, 0.01))
Q_values = tf.matmul(mapping, state)
action = tf.argmax(Q_values, 1)
next_Q_values = tf.placeholder(shape=[2,1], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_Q_values- Q_values))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
opt_op = opt.minimize(loss)

# Learning parameters
gamma = 0.99
random_exploration_rate = 0.1

# Training the network
init = tf.initialize_all_variables()
num_episodes = 2

# Create lists to contain total rewards
rewards = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        observation = env.reset()
        total_rewards = 0
        done = False
        #The Q-Network
        for j in range(5):
            # Choose an action by greedily (with e chance of random action) from the Q-network
            observation = np.reshape(observation, (4, 1))

            # TODO: the dimention of this a is wrong?
            a, Q = sess.run([action, Q_values],feed_dict={state:observation})
            print(a)
            # if np.random.uniform(0, 1) < random_exploration_rate:
            #     a[0] = env.action_space.sample()

            # Run one step simulation and get feedback
            #s1, r, done,_ = env.step(a[0])
            ##Obtain the Q' values by feeding the new state through our network
            #Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            ##Obtain maxQ' and set our target value for chosen action.
            #maxQ1 = np.max(Q1)
            #targetQ = Q
            #targetQ[0,a[0]] = r + y*maxQ1
            ##Train our network using target and predicted Q values
            #_,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            #total_rewards += r
            #s = s1
            #if done == True:
            #    #Reduce chance of random action as we train the model.
            #    e = 1./((i/50) + 10)
            #    break
        #rewards.append(total_rewards)
#print "Percent of succesful episodes: " + str(sum(rewards)/num_episodes) + "%"
