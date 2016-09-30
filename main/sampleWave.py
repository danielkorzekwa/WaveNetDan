'''
Created on Sep 21, 2016

@author: daniel
'''

import numpy as np
import tensorflow as tf
import mu_law
from WaveGraph import WaveGraph

def sampleWave(waveform,sampleNum,filterOpParams=None):
    '''Sample wave form.
    
    Args:
        waveform(np.array(int)): Initial waveform
        sampleNum(int): Number of samples to generate
        filterOpParams(ndarray[n,n]): Parameters of convolution layer

    Returns:
        np.array: Generated waveform
        
    '''
    waveGraph = WaveGraph() if filterOpParams==None else WaveGraph(filterOpParams)
             
    lastProbOp = tf.slice(waveGraph.outputProbsOp, [tf.shape(waveGraph.outputProbsOp)[0] - 1, 0], [1, 256])
    lastProbReshapedOp = tf.reshape(lastProbOp,[-1])
    
    encodedWaveform = mu_law.encode(waveform,256)
    sampleWave = encodedWaveform.tolist()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(0, sampleNum):
        
        if len(sampleWave) == 0:
            sampleWave.append(decode(np.random.randint(256)))
        else:
            lastProb = sess.run(lastProbReshapedOp, feed_dict={waveGraph.waveInput:sampleWave})
           
            predSample = np.random.choice(range(256), p=lastProb)
            print('{}:,predicted={}/{}'.format(i,predSample,lastProb[predSample]))
            sampleWave.append(predSample)
    
    sess.close()
    
    return mu_law.decode(np.array(sampleWave),256)

def decode(encodedSample):
    mu = 255
    y = encodedSample
    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1 / mu) * ((1 + mu) ** abs(y) - 1)
    return x   
