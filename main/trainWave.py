'''
Created on Sep 26, 2016

@author: daniel
'''
from WaveGraph import WaveGraph
import numpy as np
import tensorflow as tf
import mu_law
import matplotlib.pyplot as plt

PLOT_WINDOW_SIZE = 200

def trainWave(waveform, maxIterNum):
    ''' Trains wave model.
    
    Args
        waveform(np.array(int)): Waveform array of amplitudes
    
    Returns
        ndarray[n,n]: Parameters of convolution layer
    '''
    
    waveGraph = WaveGraph()
    
    encodedWaveform = mu_law.encode(waveform, 256)
        
    waveFormLen = len(waveform)
        
    entropyOp = tf.reduce_mean(
        tf.reduce_sum(
            -waveGraph.waveInputOneHotOp[1:waveFormLen, ] * tf.log(waveGraph.outputProbsOp[0:waveFormLen - 1, ]), reduction_indices=[1]))
        
    trainOp = tf.train.AdamOptimizer(0.01).minimize(entropyOp)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    for i in range(maxIterNum):
        trainOpOutput,entropy, outputOneHot = sess.run([trainOp,entropyOp, waveGraph.outputProbsOp], feed_dict={waveGraph.waveInput:encodedWaveform})
        if i % 10 == 0: 
            print('iter={}, entropy={}'.format(i, entropy))
            plotPredictedWaveForm(outputOneHot, encodedWaveform)
                            
    filterOp = sess.run(waveGraph.filterParamsVar, feed_dict={waveGraph.waveInput:encodedWaveform})
    sess.close()
    return filterOp

def plotPredictedWaveForm(outputOneHot, encodedWaveform):
    predictedWaveForm = [np.random.choice(range(256), p=x) for x in outputOneHot[0:PLOT_WINDOW_SIZE]]
                           
    plt.close()
    plt.plot(predictedWaveForm)
    plt.plot(encodedWaveform[0:PLOT_WINDOW_SIZE])
    plt.show(block=False)
            
