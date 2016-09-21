'''
Created on Sep 21, 2016

@author: daniel
'''

import tensorflow as tf

sess = tf.Session()
print('a')
t = tf.truncated_normal([1], 0, 1)
for i in range(1000):
   print(sess.run(t))
print('b')
