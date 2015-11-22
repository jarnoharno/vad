import csv
from subprocess import call
import librosa
import numpy as np
import math
try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

def combine(signal_list, noise_list, snrlist, target_rate=8000):
    for signal_file in signal_list:
        signal, srate = read_soundfile(signal_file)
        if srate != target_rate:
            signal = librosa.core.resample(signal, srate, target_rate)
        signal = rms_normalize(signal)
        for noise_file in noise_list:
            noise, nrate = read_soundfile(noise_file)
            if nrate != target_rate:
                noise = librosa.core.resample(noise, nrate, target_rate)
            if len(noise) < len(signal):
                noise = noise_tilify(noise)
                noise = rms_normalize(noise)
                noise = np.tile(noise, int(math.ceil(len(signal)/len(noise))))
            else:
                noise = rms_normalize(noise)
            for snr in snrlist:
                noisy_signal = signal*snrdb2ratio(signal)+noise
                noisy_signal = noisy_signal/peak(noisy_signal)
                new_name = soundpath+"/"+signal_file+"_"+noise_file+".flac"
                soundfile = al.Sndfile(new_name, 'r', 'flac', 1, target_rate)
                soundfile.write_frames(noisy_signal)
                soundfile.sync()

def tilify_signal(signal, rate, sfade):
    l = len(signal)/2
    head = signal[:l]
    tail = signal[l+1:]
    fade = min(l, rate*sfade)
    #faderange = np.arange(0,0.5*math.pi,1.0/fade)
    unit = 1.0/fade
    outgain = np.sin(math.pi*np.arange(0,0.5+unit, unit))
    ingain = np.cos(math.pi*np.arange(0,0.5+unit, unit))
    fadeout = head[:fade]*outgain
    fadein = tail[-fade:]*ingain
    return np.concatenate((tail[:fade], fadeout+fadein, head[fade+1:]))

def rms_normalize(signal):
    return signal/rms(signal)

def snrdb2ratio(db, signal, noise):
    return 10**(db/10)*rms(signal)/rms(noise)

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def peak(signal):
    return np.max(np.abs(signal))

def spect_power(frame, rate, size): #size=len(frame)
    k = arange(size)
    T = float(size)/rate
    frq = k/T
    frq = frq[range(size/2)]

    Y = np.fft.fft(frame)/size
    Y = Y[range(size/2)]
    return abs(Y)
