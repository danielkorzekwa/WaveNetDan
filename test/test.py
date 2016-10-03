'''
Created on Sep 21, 2016

@author: daniel
'''

import numpy as np
import tensorflow as tf
import time
from timeit import Timer
import timeit

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import wave
import loadWave
import librosa

sess = tf.Session()

a = tf.constant([[[4.0],[5.0],[6.0],[7.0],[8.0],[9.0]]])
print(a)

padding = [[0, 0], [1, 0], [0, 0]]
a_pad = tf.pad(a,padding)

f = tf.ones([3, 1, 1])

b = tf.nn.conv1d(a_pad,f,stride=1,padding='SAME')

result = tf.slice(b, [0, 0, 0], [-1, tf.shape(a)[1], -1])

print(sess.run(result))
