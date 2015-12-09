#!/usr/bin/env python
# -*- coding: utf-8 -*-
#https://gist.github.com/shunsukeaihara/4603147#file-ltsd_vad-py
import wave
import numpy as np
import scipy as sp
import speech_processing as speech
import vad_eval as vad
import librosa
try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

#WINSIZE=8192
WINMS = 50

def read_signal(filename, winsize):
    wf=wave.open(filename,'rb')
    n=wf.getnframes()
    str=wf.readframes(n)
    #channels, samplewidth?, framerate, nframes, comptype, compname
    #why?
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen=((int )(len(str)/2/winsize) + 1) * winsize
    signal=sp.zeros(siglen, sp.int16)
    signal[0:len(str)/2] = sp.fromstring(str,sp.int16)
    return [signal, params]

def get_frame(signal, winsize, no):
    shift=winsize/2
    start=no*shift
    end = start+winsize
    return signal[start:end]

class LTSD():
    def __init__(self,winsize,window,order, init_noise=None):
        self.winsize = winsize
        self.window = window
        self.order = order
        self.amplitude = {}
        self.E0 = 0
        self.E1 = 0
        self.init_noise = init_noise

    def get_amplitude(self,signal,l):
        if self.amplitude.has_key(l):
            return self.amplitude[l]
        else:
            amp = sp.absolute(sp.fft(get_frame(signal, self.winsize,l) * self.window))
            self.amplitude[l] = amp
            return amp

    def compute_noise_avg_spectrum(self,nsignal):
        windownum = len(nsignal)/(self.winsize/2) - 1
        avgamp = np.zeros(self.winsize)
        for l in xrange(windownum):
            avgamp += sp.absolute(sp.fft(get_frame(nsignal, self.winsize,l) * self.window))
        return avgamp/float(windownum)

    def compute(self,signal):
        self.windownum = len(signal)/(self.winsize/2) - 1
        ltsds = np.zeros(self.windownum)
        #Calculate the average noise spectrum amplitude based　on 20 frames in the head parts of input signal.
        #print("first frames", self.winsize,self.winsize*10,self.winsize*20)
        noise_start = self.winsize
        noise_end = self.winsize*20
        if self.init_noise is None:
            noise_magnitudes=np.zeros(9)
            noise = signal
            print("not using auxilliary noise signal")
        else:
            print("processing auxilliary noise signal")
            noise = self.init_noise*(speech.rms(signal)/speech.rms(self.init_noise))
            noise_magnitudes=np.zeros(len(noise[noise_start:noise_end])/self.winsize/2)
        for i in range(0,len(noise_magnitudes)):
            noise_magnitudes[i] = np.sum((get_frame(noise[noise_start:noise_end], self.winsize, i+1)*self.window)**2)
        self.avgnoise = self.compute_noise_avg_spectrum(noise[noise_start:noise_end])**2
        self.E0 = min(noise_magnitudes)
        self.E1 = max(noise_magnitudes)
        #print(self.E0, self.E1)
        for l in xrange(self.windownum):
            ltsds[l] = self.ltsd(signal, l, 5)
        return ltsds, np.percentile(ltsds[2:20],75), noise_start, noise_end

    def gamma(self, e):
        if e <= self.E0:
            return gamma0
        elif self.E0 < e < self.E1:
            return None #TODO
        else:
            return gamma0

    def ltse(self,signal,l,order):
        maxmag = np.zeros(self.winsize)
        for idx in range(l-order,l+order+1):
            amp = self.get_amplitude(signal,idx)
            maxmag = np.maximum(maxmag,amp)
        return maxmag

    def ltsd(self,signal,l,order):
        if l < order or l+order >= self.windownum:
            return 0
        return 10.0*np.log10(np.sum(self.ltse(signal,l,order)**2/self.avgnoise)/float(len(self.avgnoise)))

    def update_avgnoise(ltsds, l, k, alpha=0.25):
        self.avgnoise = self.compute_noise_avg_spectrum(ltsds[i-k:i])**2

    def segments(self, frames, ltsds, t, min_len=30):
        ranges = []
        segment=[]
        n_noise = 0 #n_of_noise_neighbors
        for i in range(0, len(ltsds)):
            if ltsds[i]>t:
                if len(segment) == 0 or len(segment) == 1:
                    segment.append(i)
                elif len(segment) == 2:
                    segment[1] = i
            else:
                if len(segment) == 2:
                    ranges.append(segment)
                segment = []
                self.update_avgnoise(ltsds,i,k)
                t = self.gamma()
        return ranges

def test(filename=None):
    import random, os
    import matplotlib.pyplot as plt
    from sys import argv
    #signal, params = read_signal(sound,WINSIZE)
    scenario=None
    if filename != None:
        scene = os.path.basename(filename)[0]
    else:
        filename = random.choice([x for x in os.listdir("tmp/") if os.path.splitext(x)[1] == ".flac"])
        scene = filename[0]
        filename = "tmp/"+filename
    print(filename)
    truths = vad.load_truths()
    signal,rate = speech.read_soundfile(filename)
    seconds = float(len(signal))/rate
    winsize = librosa.time_to_samples(float(WINMS)/1000, rate)[0]
    window = sp.hanning(winsize)
    ltsd = LTSD(winsize,window,5)
    res, threshold,nstart,nend =  ltsd.compute(signal)
    segments = ltsd.segments(res, threshold)
    #print(float(len(signal))/rate, librosa.core.frames_to_time(len(res), 8000, winsize/2))
    segments = librosa.core.frames_to_time(segments, rate, winsize/2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot((signal/np.max(signal))*np.mean(res)+np.mean(res))
    ax.plot(np.linspace(0,seconds, len(res)), res)
    ax.plot([0, seconds], [threshold, threshold])
    vad.plot_segments(truths[scene]['combined'], segments, ax)
    n1 = float(nstart)/rate
    n2 = float(nend)/rate
    ax.vlines([n1,n2], -20,20)
    plt.show()

def vad(soundfile, noisefile=None):
    signal,rate = speech.read_soundfile(soundfile)
    if noisefile != None:
        noise,nrate = speech.read_soundfile(noisefile)
        print("found noisefile: "+noisefile)
    else:
        noise = None
    seconds = float(len(signal))/rate
    winsize = librosa.time_to_samples(float(WINMS)/1000, rate)[0]
    window = sp.hanning(winsize)
    ltsd = LTSD(winsize,window,5, init_noise=noise)
    res, threshold,nstart,nend =  ltsd.compute(signal)
    segments,  = ltsd.segments(res, threshold)
    #print(float(len(signal))/rate, librosa.core.frames_to_time(len(res), 8000, winsize/2))
    segments = librosa.core.samples_to_time(segments, rate).tolist()
    indexes = []
    for s in segments:
        indexes += s
    indexes.append(seconds)
    return indexes


if __name__ == "__main__":
    import random, os, sys
    import matplotlib.pyplot as plt
    from sys import argv
    if len(sys.argv) >= 3:
        for f in os.listdir(argv[1]):
            if os.path.splitext(f)[1] == ".flac":
                signame = os.path.basename(os.path.splitext(f)[0])
                print(signame)
                ids = signame.split("_")
                noisefile = "noise8k/"+ids[1]+".flac"
                print(noisefile)
                #if not os.path.exists(noisefile):
                if True:
                    noisefile = None
                indexes = vad(argv[1]+f, noisefile)
                res_name = argv[2]+"/ltsd_"+os.path.basename(os.path.splitext(f)[0])+".txt"
                f = open(res_name, 'w')
                f.write("\n".join([str(x) for x in indexes]))
                f.close()
    else:
        print("Usage: "+sys.argv[0]+" [inputdir] [resultdir]")
