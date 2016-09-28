'''
Created on Sep 21, 2016

@author: daniel
'''

import numpy as np
import tensorflow as tf
import mu_law

def sampleWave(waveform,sampleNum, waveGraph,sess):
    '''Sample wave form.
    
    Args:
        waveform(np.array(int)): Initial waveform
        sampleNum(int): Number of samples to generate
        waveGraph(WaveGraph):
        sess(Session): 

    Returns:
        np.array: Generated waveform
        
    '''
             
    lastProbOp = tf.slice(waveGraph.outputProbsOp, [tf.shape(waveGraph.outputProbsOp)[0] - 1, 0], [1, 256])
    lastProbReshapedOp = tf.reshape(lastProbOp,[-1])
    
    encodedWaveform = mu_law.encode(waveform,256)
    sampleWave = encodedWaveform.tolist()
    
    for i in range(0, sampleNum):
        
        if len(sampleWave) == 0:
            sampleWave.append(decode(np.random.randint(256)))
        else:
            lastProb = sess.run(lastProbReshapedOp, feed_dict={waveGraph.waveInput:sampleWave})
           
            predSample = np.random.choice(range(256), p=lastProb)
            print('Last samples={}/{}, predicted={}/{}'.format(sampleWave[-1],lastProb[sampleWave[-1]],predSample,lastProb[predSample]))
            sampleWave.append(predSample)
    
    return mu_law.decode(np.array(sampleWave),256)

def decode(encodedSample):
    mu = 255
    y = encodedSample
    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1 / mu) * ((1 + mu) ** abs(y) - 1)
    return x   
