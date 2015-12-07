from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math, os, re
from scipy import signal, arange
from sys import argv
import sigproc as sigutil
from sklearn.preprocessing import normalize
import yin
import librosa
import vad_eval as vad
import random
import speech_processing as speech

try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

def pipeline(path, frame_ms=30, hop_ms=15, filt=True, noisy=True, shift=True):
    print("load")
    #sig, rate = librosa.load(path)
    #sig2, rate2 = ad.read_file(path)
    soundfile = al.Sndfile(path, 'r')
    rate = soundfile.samplerate
    sig = soundfile.read_frames(soundfile.nframes)
    sig = signal.wiener(sig)
    print("rate", rate)
    fsize = librosa.time_to_samples(float(frame_ms)/1000, rate)[0]
    hop = librosa.time_to_samples(float(hop_ms)/1000, rate)[0]
    print("frame size", fsize, "hop", hop)
    if filt:
        sig = bp_filter(sig)
    if noisy:
        sig = speech.add_noise(sig, "noise8k/white.flac", 30)
    frames = librosa.util.frame(sig, fsize, hop)
    w = signal.hann(fsize)
    #frames_W = np.zeros_like(frames)
    #print(frames.shape)
    #frames = frames.T
    #print(w.shape)
    print("windowing function")
    frames_w = np.apply_along_axis(lambda x,w: x*w, 0, frames, w)
    frames = frames_w
    print("window suppression")
    frames = np.apply_along_axis(lambda x,w: x/(w+1e-15), 0, frames, w)
    #    frames_W[i] = signal.convolve(frames[i],w, mode='same')
    #frames = frames_W.T
    #w = signal.correlate(w,w,mode='full')
    #w = w[w.size/2:]
    #print(frames.shape)
    #frames = sigutil.enframe(sig, fsize, hop, signal.hann)
    print("normalized autocorrelation")
    naccs = np.apply_along_axis(nacc, 0, frames)
    print("trimming")
    naccs = np.apply_along_axis(trim_frame, 0, naccs)
    minacs = np.zeros_like(naccs)
    for i in range(len(naccs.T)):
        minacs[:,i] = min_ac(naccs.T, i)
    print(naccs.shape)
    print(minacs.shape)
    print("lags")
    lags = np.zeros(len(naccs.T))
    acf_n = np.zeros(len(naccs.T))
    for i in range(len(minacs.T)):
        frame = minacs.T[i]
        relmax = signal.argrelmax(frame)[0]
        if len(relmax)>0:
            argmax2 = relmax[0] + np.argmax(frame[relmax[0]:])
        else:
            argmax2 = np.argmax(frame)
        #print(relmax)
        """
        if len(relmax)>=2:
            #print(relmax[0], relmax[1], relmax[1]-relmax[0])
            lags[i] = relmax[1]-relmax[0]
        elif len(relmax) == 1:
            lags[i] = relmax[0]
        """
        lags[i] = argmax2
        acf_n[i] = len(relmax)
        #print(lags[i], len(relmax))
        minacs.T[i] = np.roll(frame, argmax2)
    print("variances")
    #acvars = np.apply_along_axis(acvar, 0, naccs2)
    acvars = np.apply_along_axis(acvar, 0, minacs)
    print("ltacs")
    ltacs = np.zeros_like(acvars)
    for i in range(len(acvars)):
        ltacs[i] = ltac(acvars, i)
    print("done")
    return sig, rate, frames, fsize, minacs, acvars, ltacs, (lags, acf_n)

def bp_filter(data, lowcut=100.0, highcut=2000.0, fs=8000.0, order=4):
    nyq = 0.5 *fs
    low = lowcut/nyq
    high = highcut / nyq
    b,a = signal.butter(order, low, btype='highpass')
    return signal.lfilter(b, a, data)

def power_spectrum(spectrum, ref=np.max):
    return librosa.logamplitude(np.abs(spectrum)**2, ref_power=ref)

def nacc(frame):
    #print(len(frame))
    ac = ac_conv(frame)
    norm = np.sum(frame**2)
    return ac/norm

def trim_frame(frame, part=0.025):
    samples = int((len(frame)*part))
    return frame[samples:len(frame)-samples]

def min_ac(ac, l, R1=2, R2=2):
    r1 = max(0,l-R1)
    r2 = min(len(ac),l+R1)
    return np.min(ac[r1:r2], 0)

def acvar(frame):
    K=len(frame)
    Mv = 1.0/K*np.sum(frame)
    return 1.0/K*np.sum((frame-Mv)**2)

def ltac(acvars, l, R3=4, R4=2):
    r3 = max(0,l-R3)
    r4 = min(len(acvars),l+R4)
    factor = 1.0/(R3+R4+1)
    var1 = factor*np.sum(acvars[r3:r4])
    return 10*np.log10(factor*np.sum((acvars[r3:r4]-var1)**2))


def autocorrelation(rosa_frames):
    frames = rosa_frames
    ac2 = np.apply_along_axis(ac_conv, 1, frames)
    A = librosa.logamplitude(ac2.T, ref_power=np.max)
    return ac2, A

# autocorrelation by convolution
def ac_conv(frame):
    acorr = np.correlate(frame,frame,mode='full')
    return acorr[acorr.size/2:]


# statistical autocorrelation ... incomplete
def ac_stat(frame):
    acorr = np.correlate(frames,f,mode='full')
    return acorr[acorr.size/2:]

def predict(signal, threshold=-55, frame_hop=120):
    ranges = []
    segment=[]
    for i in range(0, len(signal)):
        if threshold_test(signal,threshold,i):
            if len(segment) == 0 or len(segment) == 1:
                segment.append(i)
            elif len(segment) == 2:
                segment[1] = i
        else:
            if len(segment) == 2:
                ranges.append(segment)
            segment = []
    segments = librosa.core.frames_to_time(ranges, rate, frame_hop).tolist()
    return segments

def write_results(segments, res_name, l):
    indexes = []
    for s in segments:
        indexes += s
    indexes.append(l)
    f = open(res_name, 'w')
    f.write("\n".join([str(x) for x in indexes]))
    f.close()

def threshold_test(s,t,i):
    if hasattr(t, "__len__"):
        return s[i]>t[i]
    else:
        return s[i]>t

def local_min_array(x, W_len=320):
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

def moving_average(x, W_len=320):
    """ Get moving average of signal """
    #frame_ms = 10 #how many ms is one frame 
    w = np.ones(W_len,'d')
    padd_y = np.abs(np.amin(x))
    a = np.convolve(w/w.sum(),x+padd_y,mode='same')
    return a-padd_y

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from sys import argv
    #signal, params = read_signal(sound,WINSIZE)
    scenario=None
    truths = vad.load_truths()
    args = set(['sig', 'ac-spec', 'var', 'ltac', 'ac-feature', 'batch'])
    if len(argv) >= 2 and argv[1] in args:
        if len(argv)>=3 and argv[1] != 'batch':
            filename = argv[2]
            scene = os.path.basename(filename)[0]
        else:
            filename = random.choice([x for x in os.listdir("tmp/") if os.path.splitext(x)[1] == ".flac"])
            scene = filename[0]
            filename = "tmp/"+filename
            print(filename)
        if argv[1] != 'batch':
            sig, rate, frames, fsize, naccs, acvars, ltacs, more = pipeline(filename)
            predictions = []
            seconds = float(len(sig))/rate
        if argv[1] == 'sig':
            plt.plot(sigutil.deframesig(frames.T,len(sig),fsize,fsize/2,signal.hanning))
            plt.show()
        elif argv[1] == 'ac-spec':
            librosa.display.specshow(naccs)
            plt.show()
        elif argv[1] == 'var':
            vad.plot_segments(truths[scene]['combined'], [], plt)
            plt.plot(np.linspace(0,seconds, len(acvars)), acvars)
            plt.show()
        elif argv[1] == 'ltac':
            vad.plot_segments(truths[scene]['combined'], [], plt)
            plt.plot(np.linspace(0,seconds, len(ltacs)), ltacs)
            plt.show()
        elif argv[1] == 'ac-feature':
            vad.plot_segments(truths[scene][scene+'i'], 'ti', plt)
            vad.plot_segments(truths[scene][scene+'j'], 'tj', plt)
            lmin, smoothmin = local_min_array(ltacs)
            lmin = lmin+7
            vad.plot_segments(predict(ltacs, lmin), 'p', plt)
            #plt.plot(np.linspace(0,seconds, len(acvars)), acvars)
            plt.plot(np.linspace(0,seconds, len(ltacs)), ltacs)
            plt.plot(np.linspace(0,seconds, len(lmin)), lmin+2)
            #plt.plot(np.linspace(0,seconds, len(more[0])), more[0])
            plt.plot(np.linspace(0,seconds, len(more[1])), more[1])
            plt.show()
        elif argv[1] == 'batch':
            for f in os.listdir(argv[2]):
                if os.path.splitext(f)[1] == ".flac":
                    signame = os.path.basename(os.path.splitext(f)[0])
                    print(signame)
                    ids = signame.split("_")
                    print(argv[2]+f)
                    sig, rate, frames, fsize, naccs, acvars, ltacs, more = pipeline(argv[2]+f)
                    seconds = float(len(sig))/rate
                    lmin, smoothmin = local_min_array(ltacs)
                    lmin = lmin+7
                    segments = predict(ltacs, lmin)
                    res_name = argv[3]+"/acc_"+os.path.basename(os.path.splitext(f)[0])+".txt"
                    write_results(segments, res_name, seconds)
    else:
        print("usage "+argv[0]+" <<"+("|".join(args))+"> [soundfile] | batch sfpath respath>")
