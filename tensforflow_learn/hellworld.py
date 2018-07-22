#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from scipy import signal as sg

"""
Simple tensorflow workflow
"""
a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(c)
    print(result)

"""
Convolution
"""

x = [6,2]
h = [1,2,5,4]

# 1D convolution with zeros paddings
y = np.convolve(x, h, "full")
print(h)

# 1D convolution without paddings
y = np.convolve(x, h, "valid")
print(y)

# 2D convolution 

I = [[255, 7, 3],
     [212, 240, 4],
     [218, 216, 230]]

g = [[-1, 1]]

print(sg.convolve(I, g))
