'''
Created on Sep 21, 2016

@author: daniel
'''

import numpy as np
import tensorflow as tf
import time
from timeit import Timer
import timeit

sess = tf.Session()


a = tf.constant([[1,2,3],[4,5,6]])
b = a[1]
c = tf.slice(a,[1,0],[1,3])

print(b,c)
print(sess.run(b))
print(sess.run(c))


