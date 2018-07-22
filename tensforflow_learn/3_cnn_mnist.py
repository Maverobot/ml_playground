#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
width = 28
height = 28
flat = width * height
class_output = 10

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, flat])
y = tf.placeholder(tf.float32, shape=[None, class_output])
keep_prob = tf.placeholder(tf.float32)

# Reshape input x to image
x_image = tf.reshape(x, [-1, 28, 28, 1])

"""
CNN model definition
"""

# CNN layer 1
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1
h_conv1 = tf.nn.relu(convolve1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# CNN layer 2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# Fully connected layer 1
conv2_matrix = tf.reshape(conv2, [-1, 7*7*64])
W_fcl1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fcl1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fcl1 = tf.matmul(conv2_matrix, W_fcl1) + b_fcl1
h_fcl1 = tf.nn.relu(h_fcl1)
fcl1 = tf.nn.dropout(h_fcl1, keep_prob)

# Fully connected layer 2
W_fcl2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fcl2 = tf.Variable(tf.constant(0.1, shape=[10]))
fcl2 = tf.matmul(fcl1, W_fcl2) + b_fcl2
y_cnn = tf.nn.softmax(fcl2)

"""
Training specification
"""

# Cost and optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_cnn), axis=1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Prediction
correct_prediction = tf.equal(tf.argmax(y_cnn, axis=1), tf.argmax(y, axis=1))
# Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Training, testing and evaluation
"""

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for i in range(500):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0}) * 100
        print("step %d, training accuracy %f %%" % (i, train_accuracy))

test_batch = mnist.test.next_batch(200)
test_accuracy = accuracy.eval(feed_dict={x:test_batch[0], y:test_batch[1], keep_prob:1.0}) * 100
print("test accuracy %f %%" % test_accuracy)

sess.close()
