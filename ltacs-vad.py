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

def pipeline(path, frame_ms=30, hop_ms=15):
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
    print(naccs.shape)
    minacs = np.zeros_like(naccs)
    for i in range(len(naccs.T)):
        minacs[:,i] = min_ac(naccs.T, i)
    print(minacs.shape)
    print("variances")
    #acvars = np.apply_along_axis(acvar, 0, naccs2)
    acvars = np.apply_along_axis(acvar, 0, minacs)
    print("ltacs")
    ltacs = np.zeros_like(acvars)
    for i in range(len(acvars)):
        ltacs[i] = ltac(acvars, i)
    return sig, rate, frames, fsize, minacs, acvars, ltacs

# autocorrelation by convolution
def ac_conv(frame):
    acorr = np.correlate(frame,frame,mode='full')
    return acorr[acorr.size/2:]

def nacc(frame):
    #print(len(frame))
    ac = ac_conv(frame)
    norm = np.sum(frame**2)
    return ac/norm

def trim_frame(frame, part=0.025):
    samples = int((len(frame)*part))
    return frame[samples:len(frame)-samples]

def min_ac(ac, l, R1=4, R2=4):
    r1 = max(0,l-R1)
    r2 = min(len(ac),l+R1)
    return np.min(ac[r1:r2], 0)

def acvar(frame):
    K=len(frame)
    Mv = 1.0/K*np.sum(frame)
    return 1.0/K*np.sum((frame-Mv)**2)

def ltac(acvars, l, R3=10, R4=10):
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


# statistical autocorrelation ... incomplete
def ac_stat(frame):
    acorr = np.correlate(frames,f,mode='full')
    return acorr[acorr.size/2:]

def write_results(segments, res_name, l):
    indexes = []
    for s in segments:
        indexes += s
    indexes.append(l)
    f = open(res_name, 'w')
    f.write("\n".join([str(x) for x in indexes]))
    f.close()

def compute_vad(args):
    filename, path, resultpath = args
    signame = os.path.basename(os.path.splitext(filename)[0])
    ids = signame.split("_")
    print("computing: "+path+filename)
    sig, rate, frames, fsize, nacc, acvars, ltacs = pipeline(path+filename)
    hop = fsize/2
    seconds = float(len(sig))/rate
    print(hop)
    segments,thresholds = predict(ltacs, rate=rate, frame_hop=hop)
    res_name = resultpath+"/ltacs_"+os.path.basename(os.path.splitext(filename)[0])+".txt"
    write_results(segments, res_name, seconds)

def predict(signal, alpha=0.25, rate=8000, frame_hop=120):
    beta = .95
    khi_sn = np.zeros(100)
    khi_n = np.repeat(np.min(signal[:13]), 100)
    khi_n[:13] = signal[:13]
    mu_n = np.min(signal[:13])
    w_n = np.max(signal[:13])
    print(len(khi_n))
    ranges = []
    segment=[]
    threshold = mu_n + beta*(w_n-mu_n)
    computed_thresholds = np.zeros(len(signal))
    for i in range(0, len(signal)):
        computed_thresholds[i] = threshold
        if signal[i] > threshold:
            khi_sn=np.roll(khi_sn, -1)
            khi_sn[-1] = signal[i]
            if len(segment) == 0 or len(segment) == 1:
                segment.append(i)
            elif len(segment) == 2:
                segment[1] = i
        else:
            khi_n=np.roll(khi_n, -1)
            khi_n[-1] = signal[i]
            if len(segment) == 2:
                ranges.append(segment)
            segment = []
        if i > 100:
            threshold = alpha*np.min(khi_sn)+(1-alpha)*np.max(khi_n)
        else:
            mu_n = np.mean(signal[:max(13,i)])
            w_n = np.max(signal[:max(13,i)])
            threshold = mu_n + beta*(w_n-mu_n)
    segments = librosa.core.frames_to_time(ranges, rate, frame_hop).tolist()
    return segments, computed_thresholds

if __name__ == "__main__":
    import random, os
    import matplotlib.pyplot as plt
    from sys import argv
    #signal, params = read_signal(sound,WINSIZE)
    scenario=None
    if len(argv)==3 and argv[1] is not 'batch':
        filename = argv[2]
        scene = os.path.basename(filename)[0]
        truths = vad.load_truths()
        print(filename)
        sig, rate, frames, fsize, nacc, acvars, ltacs = pipeline(filename)
        seconds = float(len(sig))/rate
    elif len(argv) < 3:
        filename = random.choice([x for x in os.listdir("tmp/") if os.path.splitext(x)[1] == ".flac"])
        scene = filename[0]
        filename = "tmp/"+filename
        truths = vad.load_truths()
        print(filename)
        sig, rate, frames, fsize, nacc, acvars, ltacs = pipeline(filename)
        seconds = float(len(sig))/rate
    if argv >= 2 and argv[1] is not 'batch':
        if argv[1] == 'sig':
            plt.plot(sigutil.deframesig(frames.T,len(sig),fsize,fsize/2,signal.hanning))
            plt.show()
        if argv[1] == 'ac':
            librosa.display.specshow(nacc)
            plt.show()
        elif argv[1] == 'var':
            vad.plot_segments(truths[scene][scene+'i'], 'ti', plt)
            vad.plot_segments(truths[scene][scene+'j'], 'tj', plt)
            plt.plot(np.linspace(0,seconds, len(acvars)), acvars)
            plt.show()
        elif argv[1] == 'ltac':
            vad.plot_segments(truths[scene][scene+'i'], 'ti', plt)
            vad.plot_segments(truths[scene][scene+'j'], 'tj', plt)
            plt.plot(np.linspace(0,seconds, len(ltacs)), ltacs)
            plt.show()
        elif argv[1] == 'test':
            print(len(ltacs))
            segments,thresholds = predict(ltacs)
            vad.plot_segments(truths[scene][scene+'i'], 'ti', plt)
            vad.plot_segments(truths[scene][scene+'j'], 'tj', plt)
            vad.plot_segments(segments, 'p', plt)
            plt.plot(np.linspace(0,seconds, len(ltacs)), ltacs)
            plt.plot(np.linspace(0,seconds, len(thresholds)), thresholds)
            plt.show()
    if len(argv) > 3 and argv[1] == 'batch':
        files = []
        for f in os.listdir(argv[2]):
            if os.path.splitext(f)[1] == ".flac":
                files.append(f)
        pool = multiprocessing.Pool(None)
        args = [(f, argv[2], argv[3]) for f in files]
        r = pool.map_async(compute_vad, args)
        r.wait()
        #for arg in args:
        #    compute_vad(arg)
    print(argv)
