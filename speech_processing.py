import csv
from subprocess import call
import librosa
import numpy as np
import math
import os
try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

def combine(signal_list, noise_list, snrlist, soundpath="tmp", target_rate=8000, overwrite=True):
    for signal_file in signal_list:
        signal, srate = read_soundfile(signal_file)
        if srate != target_rate:
            signal = librosa.core.resample(signal, srate, target_rate)
        signal = rms_normalize(signal)
        print("Opened ", signal_file)
        print("signal", signal.shape)
        for noise_file in noise_list:
            noise, nrate = read_soundfile(noise_file)
            print("noise", noise.shape)
            if nrate != target_rate:
                noise = librosa.core.resample(noise, nrate, target_rate)
            if len(noise) < len(signal):
                print("tiling")
                noise = tilify_signal(noise, nrate, 0.5)
                noise = rms_normalize(noise)
                noise = np.tile(noise, int(math.ceil(float(len(signal))/len(noise))))[:len(signal)]
            else:
                noise = rms_normalize(noise)[:len(signal)]
            print("Opened ", noise_file)
            for snr in snrlist:
                signal_name = os.path.basename(os.path.splitext(signal_file)[0])
                noise_name = os.path.basename(os.path.splitext(noise_file)[0])
                new_name = soundpath+"/"+signal_name+"_"+noise_name+"_"+str(snr)+".flac"
                print(new_name)
                if overwrite or os.path.exists(new_name) == False:
                    print("Combining with SNR", snr)
                    noisy_signal = signal*snrdb2ratio(snr, signal, noise)+noise
                    noisy_signal = noisy_signal/peak(noisy_signal)
                    soundfile = al.Sndfile(new_name, 'w', al.Format('flac'), 1, target_rate)
                    soundfile.write_frames(noisy_signal)
                    soundfile.sync()
                    print("Wrote", new_name)
                else:
                    print(new_name+" exists, skipping")

def tilify_signal(signal, rate, sfade):
    l = len(signal)/2
    head = signal[:l]
    tail = signal[l+1:]
    fade = min(l, rate*sfade)
    #faderange = np.arange(0,0.5*math.pi,1.0/fade)
    unit = 1.0/fade
    outgain = np.sin(math.pi*np.arange(0,0.5+unit, fade))
    ingain = np.cos(math.pi*np.arange(0,0.5+unit, fade))
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

def read_soundfile(path):
    if al != None:
        soundfile = al.Sndfile(path, 'r')
        return soundfile.read_frames(soundfile.nframes), soundfile.samplerate
    else:
        try:
            print("Warning: no audiolab. Trying to read WAV: "+path)
            wav = wavfile.read(path)[1]
            wav = np.float64(wav)/np.iinfo(np.int16).max
            return wav
        except ValueError:
            return None
