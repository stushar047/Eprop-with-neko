import numpy as np
import math
from matplotlib import pyplot as plt
class encoder():
    def __init__(self,wx=5,wy=5,potx=28,poty=28,rv=2,ox=2,oy=2):
        self.wx=wx
        self.wy=wy
        self.potx=potx
        self.poty=poty
        self.rv=rv
        self.ox=ox
        self.oy=oy
    def rate_encoding(self,img):
    # Sliding window implementation of receptive field
        w = np.zeros([self.wx,self.wy])
        pot = np.zeros([self.potx//2,self.poty//2])
        ran = list(range(-self.rv,self.rv+1))
        ox = self.ox
        oy = self.oy
        w[ox][oy] = 1

        for i in range(self.wx):
            for j in range(self.wy):
                d = abs(ox-i) + abs(oy-j)
                w[i][j] = (-0.375)*d + 1

        #reading dataset image (16x16)

        #calculating potential map of the image (256 input neuron potential)
        for i in range(self.potx//2):
            for j in range(self.poty//2):
                summ = 0
                for m in ran:
                    for n in ran:
                        if (i+m)>=0 and (i+m)<=15 and (j+n)>=0 and (j+n)<=15:
                            summ = summ + w[ox+m][oy+n]*img[i+m][j+n]
                pot[i][j] = summ

        #defining time frame of 1s with steps of 5ms
        T = 1;
        dt = 0.005
        time  = np.arange(0, T+dt, dt)

        #initializing spike train
        train = []

        for l in range(self.potx//2):
            for m in range(self.poty//2):
                temp = np.zeros([201,])
        #calculating firing rate proportional to the membrane potential
                freq = math.ceil(0.102*pot[l][m] + 52.02)
                if freq<=0:
                    freq=0.0001;
                else:
                    freq=freq
                freq1 = math.ceil(200/freq)
#                 print(freq)

                #generating spikes according to the firing rate
                k = 0
                while k<200:
                    temp[k] = 1
                    k = k + freq1
                train.append(temp) 
        return train

    def create_Encoded_Dataset(self,x):
        x_enc=[self.rate_encoding(x[i]) for i in range(len(x))]
        self.x_enc=x_enc
        