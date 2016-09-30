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


waveform = librosa.load('/home/daniel/daniel/test.wav',7000)    