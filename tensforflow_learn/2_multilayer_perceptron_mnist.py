#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Interacive Session, i.e. blabla.eval() is possible
sess = tf.Session()

# Input and output tensor data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight tensor
W = tf.Variable(tf.zeros([784, 10]), tf.float32)

# Bias tensor
b = tf.Variable(tf.zeros([10]), tf.float32)

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Training
costs = []
sess.run(tf.global_variables_initializer())
for i in range(500):
    batch = mnist.train.next_batch(50)
    _, cost = sess.run([train_step, cross_entropy], feed_dict={x:batch[0], y_:batch[1]})
    costs.append(cost)

# Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}, session=sess) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

# Plot cost convergence
plt.xlabel("iterations")
plt.ylabel("crossentropy")
plt.plot(costs)
plt.show()

# Close the session
sess.close()

