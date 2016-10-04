'''
Created on Oct 4, 2016

@author: daniel
'''
import librosa
import matplotlib.pyplot as plt

wavData = librosa.load('/home/daniel/daniel/do.wav',8000)[0]
wavData2 = librosa.load('/home/daniel/daniel/test.wav',8000)[0]
print(wavData)
#plt.plot(wavData[0:100])
a=4000
b=8000
plt.plot(range(a,b),wavData[a:b],label='test')
plt.plot(range(a,b),wavData2[a:b],label='predicted')
plt.legend()
plt.show()
