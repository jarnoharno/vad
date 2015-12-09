from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math, os, re
from scipy import signal, arange
from sys import argv
import sigproc as sigutil
from sklearn.preprocessing import normalize
import librosa
import vad_eval as vad
import random
import speech_processing as speech
import multiprocessing

try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

#Relative spectral entropy, using running average for mean spectrum
#Soundsense version
#Todo: units: dB, temporal
def spectral_entropy(frames, samplerate=8000, frame_size=64, tail=0.8):
    #signal = smooth_signal(signal, 10) #initial smoothing of signal with a 10 frame win
    nframes = len(frames.T)
    n = len(frames)
    #p[0] = np.linalg.norm(spect_power(y[0], samplerate, n))
    print("normalize spectrum")
    #p = np.apply_along_axis(normalized_spectrum, 0, frames, samplerate)
    w = np.hanning(n)
    #windowed = np.apply_along_axis(signal.convolve, 0, frames, w, 'same')
    windowed = np.apply_along_axis(lambda x,y:x*y, 0, frames, w)
    p = np.apply_along_axis(spect_power, 0, windowed, samplerate, n)
    p = np.apply_along_axis(normalize_spectrum, 0, p, samplerate)
    print("calculate entropy")
    H = np.apply_along_axis(entropy, 0, p)
    return H,p

def normalize_spectrum(spectrum, samplerate=8000):
    norm = np.linalg.norm(spectrum) #slow
    if norm == 0: #Q: can I do this?
        #res.append(np.zeros_like(spectrum)+0.0001)
        return spectrum+1e-15
    else:
        return spectrum/norm
    #return np.sum([p[i]*np.log(p[i]))

def spect_power(frame, rate, size): #size=len(frame)
    k = arange(size)
    T = float(size)/rate
    frq = k/T
    frq = frq[range(size/2)]

    Y = np.fft.fft(frame)/size
    Y = Y[range(size/2)]
    return abs(Y)

def entropy(p):
    return -np.sum(p*np.log(p))

def average(x, W_len=60):
    """ Get moving average of signal """
    #frame_ms = 10 #how many ms is one frame 
    w = np.ones(W_len,'d')
    padd_y = np.abs(np.amin(x))
    a = np.convolve(w/w.sum(),x+padd_y,mode='same')
    return a-padd_y

def RSE(frames, samplerate=8000, frame_size=25, tail=0.8):
    #signal = smooth_signal(signal, 10) #initial smoothing of signal with a 10 frame win
    nframes = len(frames.T)
    n = len(frames)
    #p[0] = np.linalg.norm(spect_power(y[0], samplerate, n))
    print("normalize spectrum")
    p = np.apply_along_axis(normalized_spectrum, 0, frames, samplerate)
    m = np.zeros_like(p).T
    rse = np.zeros(nframes)
    m[0] = p.T[0]
    print(len(p.T))
    for t in range(1,len(p.T)):
        m[t] = m[t-1]*tail + p.T[t] * (1-tail)
    m=m.T
    for t in range(1,len(rse)):
        rse[t] = np.sum(p.T[t]*np.log(m.T[t-1]/p.T[t]))
    return rse,p,m

#RSE spectrum frames
def normalized_spectrum1(frames, samplerate=8000):
    w = np.hanning(len(frames[0]))
    res = []
    for i,frame in enumerate(frames):
        windowed = signal.convolve(frame, w, mode='same')
        spectrum = spect_power(windowed, samplerate, len(frame))
        norm = np.linalg.norm(spectrum) #slow
        if norm == 0: #Q: can I do this?
            #res.append(np.zeros_like(spectrum)+0.0001)
            res.append(spectrum+0.0001)
        else:
            res.append(spectrum/norm)
    return np.asarray(res)
    #return np.sum([p[i]*np.log(p[i]))

def local_min_array(x, W_len=60):
    """ local minimums collected into a numpy array, plus a smoothed version """
    m = signal.argrelmin(x, order=W_len/2)[0]
    j=0
    lmin = np.zeros_like(x)
    for i in range(len(x)):
        if i > m[j] and j<len(m)-1:
            j += 1
        if j == 0 or j == (len(m)-1):
            min_pos = m[j]
        elif abs(m[j]-i) < abs(m[j-1]-i):
            min_pos = m[j]
        else:
            min_pos = m[j-1]
        lmin[i] = x[min_pos]
    window_len = W_len*2
    w = np.hamming(window_len)
    lm2 = np.r_[lmin[window_len-1:0:-1], lmin, lmin[-1:-window_len:-1]]
    min_smooth = np.convolve(w/w.sum(),lm2,mode='valid')
    return lmin, min_smooth

def pipeline(path, frame_ms=64, hop_ms=64):
    sig, rate = speech.read_soundfile(path)
    fsize = librosa.time_to_samples(float(frame_ms)/1000, rate)[0]
    hop = librosa.time_to_samples(float(hop_ms)/1000, rate)[0]
    frames = librosa.util.frame(sig, fsize, hop)
    rms = np.apply_along_axis(speech.rms, 0, frames)
    H, p = spectral_entropy(frames, rate, fsize)
    return sig, rate, frames, fsize, rms, H, p

def predict(rms, H, rms_t, H_t):
    ranges = []
    segment=[]
    for i in range(0, min(len(rms), len(H))):
        if rms[i] > rms_t[i] or H[i] < H_t[i]:
            if len(segment) == 0 or len(segment) == 1:
                segment.append(i)
            elif len(segment) == 2:
                segment[1] = i
        else:
            if len(segment) == 2:
                ranges.append(segment)
            segment = []
    return ranges

def write_results(segments, res_name, l):
    indexes = []
    for s in segments:
        indexes += s
    indexes.append(l)
    f = open(res_name, 'w')
    f.write("\n".join([str(x) for x in indexes]))
    f.close()

def compute_vad(args):
    rms, H, rms_t, H_t, rate, fsize, res_name, seconds = args
    predictions = predict(rms, H, rms_t, H_t)
    predictions = librosa.core.frames_to_time(predictions, rate, fsize).tolist()
    print("SoundSense writing: "+res_name)
    write_results(predictions, res_name, seconds)

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from sys import argv
    #signal, params = read_signal(sound,WINSIZE)
    scenario=None
    truths = vad.load_truths()
    if len(argv)>=2 and argv[1] != 'batch':
        filename = argv[1]
        scene = os.path.basename(filename)[0]
    elif len(argv) == 1:
        filename = random.choice([x for x in os.listdir("tmp/") if os.path.splitext(x)[1] == ".flac"])
        scene = filename[0]
        filename = "tmp/"+filename
    if len(argv) >= 3:
	tasks = []
        pool = multiprocessing.Pool(10)
        args = [(f, argv[2], argv[3]) for f in files]
        for f in os.listdir(argv[2]):
            if os.path.splitext(f)[1] == ".flac":
                signame = os.path.basename(os.path.splitext(f)[0])
                print(signame)
                ids = signame.split("_")
                print(argv[2]+f)
                res_name = argv[3]+"/sosens_"+os.path.basename(os.path.splitext(f)[0])+".txt"
                frame_ms = 64
                sig, rate, frames, fsize, rms, H, p = pipeline(argv[2]+f, frame_ms)
                seconds = float(len(sig))/rate
                rms_t, rms_t_smooth = local_min_array(rms)
                H_a = average(H, 20)
                H_min, H_min_smooth = local_min_array(H, 30)
                rms_t += 0.012
                H_t = H_min+(H_a*0.2)
		tasks.append([rms, H, rms_t, H_t, rate, fsize, res_name, seconds])
                #predictions = predict(rms, H, rms_t, H_t)
                #predictions = librosa.core.frames_to_time(predictions, rate, fsize).tolist()
                #write_results(predictions, res_name, seconds)
        r = pool.map_async(compute_vad, tasks)
        r.wait()
	pool.close()
	pool.join()
	
    else:
        print(filename)
        frame_ms = 64
        sig, rate, frames, fsize, rms, H, p = pipeline(filename, frame_ms)
        seconds = float(len(sig))/rate
        vad.plot_segments(truths[scene][scene+'i'], 'ti', plt)
        vad.plot_segments(truths[scene][scene+'j'], 'tj', plt)
        rms_t, rms_t_smooth = local_min_array(rms)
        H_a = average(H, 20)
        H_min, H_min_smooth = local_min_array(H, 30)
        rms_t += 0.012
        H_t = H_min+(H_a*0.2)
        predictions = predict(rms, H, rms_t, H_t)
        predictions = librosa.core.frames_to_time(predictions, rate, fsize).tolist()
        vad.plot_segments(predictions, 'p', plt)
        plt.plot(np.linspace(0,seconds, len(rms)), rms)
        plt.plot(np.linspace(0,seconds, len(H)), H)
        plt.plot(np.linspace(0,seconds, len(rms_t)), rms_t)
        plt.plot(np.linspace(0,seconds, len(H_t)), H_t)
        plt.show()
