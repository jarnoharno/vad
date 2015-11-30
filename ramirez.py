# fs: sampling rate
# Nw: window size
# Nsh: window shift size
# win: windowing function (hamming)
# M: number of neighbour frames each side used to calculate LTSE

# noise updating:

# N:  average noise spectrum
# E0: energy of the cleanest condition
# E1: energy of the noisiest condition
# gamma0: low energy threshold
# gamma1: high energy threshold
# K: number of neighbor frames (each side) for updating noise spectrum
# alpha: noise spectrum updating rate

import numpy as np
import librosa
from scipy import signal
import sigutil
import speech_processing as sp
import math

class Ramirez:
    def __init__(NFFT, fs, Nw, Nsh, n, M, K=3):
        self.NFFT = NFFT
        self.fs = fs
        self.Nw = Nw
        self.Nsh = Nsh
        self.win = signal.hamming(Nw)
        self.K = K
        self.alpha = 0.95
        n = sigutil.framesig(n, Nw, Nsh, self.win)
        N = np.apply_along_axis(sp.spect_power, 1, n, fs, NFFT)
        self.N = np.mean(N[:K*2], 1) # Initial mean noise spectrum.
        energy = np.sum(n[:K*2]**2, 0)
        self.E0 = min(energy)
        self.E1 = max(energy)
        self.M = M

    def ltsd(self, s):
        x = sigutil.framesig(s, self.Nw, self.Nsh, self.win)
        X = np.apply_along_axis(sp.spect_power, 1, x, fs, Nw)

        L = len(X)
        d = np.zeros(len(l))
        flag = np.zeros(len(l))
        for l in range(self.M, L-self.M):
            ltse = np.amax(X[:,l-self.M:l+self.M],1)
            d[l] = 10*math.log10(ltse**2 * (1.0/self.N**2)/self.NFFT)

    def gamma(self, e):
        if e <= self.E0:
            return self.gamma0
        elif e >= self.E1:
            return self.gamma1
        else:
            return self.gamma0 -
                (self.gamma0-self.gamma1)/(1-self.E1/self.E0) +
                (self.gamma0-self.gamma1)/(self.E0-self.E1)*e

    def updateN(self, X, l):
        N = np.sum(X[:,l-self.K:l+self.K],0)/(2*self.K+1)
        self.N = self.alpha*self.N + (1-self.alpha)*N
