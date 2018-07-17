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
state = tf.placeholder(shape=[1,4], dtype=tf.float32)
mapping = tf.Variable(tf.random_uniform([4, 2], 0, 0.01))
actions_weights = tf.matmul(state, mapping)
action = tf.argmax(actions_weights, 1)
next_actions_weights = tf.placeholder(shape=[1,2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_actions_weights - actions_weights))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
opt_op = opt.minimize(loss)

# Learning parameters
gamma = 0.99
random_exploration_rate = 0.1

# Training the network
init = tf.initialize_all_variables()

"""
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"
"""
