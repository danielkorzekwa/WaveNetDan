'''
Created on Sep 21, 2016

@author: daniel
'''
from math import sin, pi
import tensorflow as tf


class WaveNetModel:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.i=0
        self.sess=tf.Session()
        self.normal=tf.truncated_normal([1], 0, 1000)
        
    def sample(self):
        '''Returns a single sample of a waveform
        '''
        #return int(20000*sin(self.i*(2*pi/3))) #+ random.randint(-4000, 4000)
        return self.sess.run(self.normal)[0]*0.0001
    
    def add(self,sample):
        '''Add sample to a model
        '''
        self.i+=1
        