'''
Created on Sep 26, 2016

@author: daniel
'''
from WaveGraph import WaveGraph
import numpy as np
import tensorflow as tf
import mu_law
import matplotlib.pyplot as plt

def trainWave(waveform,maxIterNum):
    ''' Trains wave model.
    
    Args
        waveform(np.array(int)): Waveform array of amplitudes
    
    Returns
        ndarray[n,n]: Parameters of convolution layer
    '''
    
    waveGraph = WaveGraph()
    
    encodedWaveform = mu_law.encode(waveform,256)
        
    entropyOp = tf.reduce_mean(tf.reduce_sum(-waveGraph.waveInputOneHotOp[1:101,] * tf.log(waveGraph.outputProbsOp[0:100,]), reduction_indices=[1]))
        
    trainOp = tf.train.GradientDescentOptimizer(0.5).minimize(entropyOp)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    for i in range(maxIterNum):
        train,entropy,outputOneHot =  sess.run([trainOp,entropyOp,waveGraph.outputProbsOp],feed_dict={waveGraph.waveInput:encodedWaveform})
        output = [np.random.choice(range(256),p=x) for x in outputOneHot[0:100]]
        if i % 10==0: 
            print('iter={}, entropy={}'.format(i,entropy))
                        
            plt.close()
            plt.plot(np.arange(0,100),output[0:100])
            plt.plot(np.arange(0,100),encodedWaveform[0:100])
            plt.show(block=False)
            
        #print(sess.run(waveTrainGraph.W),sess.run(waveTrainGraph.b))
    
    filterOp = sess.run(waveGraph.filterOp,feed_dict={waveGraph.waveInput:encodedWaveform})
    sess.close()
    return filterOp
