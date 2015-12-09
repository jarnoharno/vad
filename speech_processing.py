import csv
from subprocess import call
import librosa
import numpy as np
import math
import os
import soundfile
import vad_eval as vad
import random
import multiprocessing

try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    #print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

def combine(signal_list, noise_list, snrlist, soundpath="tmp", target_rate=8000, overwrite=False, parallel=True):
    for signal_file, labelfile in signal_list:
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
            tasks = []
            for snr in snrlist:
                signal_name = os.path.basename(os.path.splitext(signal_file)[0])
                noise_name = os.path.basename(os.path.splitext(noise_file)[0])
                new_name = soundpath+"/"+signal_name+"_"+noise_name+"_"+str(snr)+".flac"
                print(new_name)
                if overwrite or os.path.exists(new_name) == False:
                    print("Combining with SNR", snr)
                    args = [snr, signal, noise, target_rate, new_name]
                    if parallel:
                        tasks.append(args)
                    else:
                        compute_combinations(args)
                        """
                        noisy_signal = signal*snrdb2ratio(snr, signal, noise)+noise
                        noisy_signal = noisy_signal/peak(noisy_signal)
                        soundfile = al.Sndfile(new_name, 'w', al.Format('flac'), 1, target_rate)
                        soundfile.write_frames(noisy_signal)
                        soundfile.sync()
                        """
                else:
                    print(new_name+" exists, skipping")
                if parallel:
                    pool = multiprocessing.Pool(4)
                    r = pool.map_async(compute_combination, tasks)
                    r.wait()
		    pool.terminate()

def compute_combination(args):
    snr, signal, noise, target_rate, new_name = args
    noisy_signal = signal*snrdb2ratio(snr, signal, noise)+noise
    noisy_signal = noisy_signal/peak(noisy_signal)
    soundfile.write(new_name, noisy_signal, target_rate)
    #soundfile = al.Sndfile(new_name, 'w', al.Format('flac'), 1, target_rate)
    #soundfile.write_frames(noisy_signal)
    #soundfile.sync()
    print("Wrote", new_name)

def add_noise(signal, noisefile, snr=10):
    noise,nrate = read_soundfile(noisefile)
    noise = np.roll(noise, random.randint(0,len(noise)))
    if len(noise) < len(signal):
        noise = tilify_signal(noise, nrate, 0.5)
        noise = rms_normalize(noise)
        noise = np.tile(noise, int(math.ceil(float(len(signal))/len(noise))))[:len(signal)]
    else:
        noise = rms_normalize(noise)[:len(signal)]
    noisy_signal = signal*snrdb2ratio(snr, signal, noise)+noise
    noisy_signal = noisy_signal/peak(noisy_signal)
    return noisy_signal

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

#Used in noise approximation algorithms
#TODO: Actually produces corrupt data!
def resample_and_normalize_file(source, target, new_rate):
    signal, srate = read_soundfile(source)
    if srate != new_rate:
        signal = librosa.core.resample(signal, srate, new_rate)
    sig = rms_normalize(signal)
    sounfile.write(target, sig, new_rate)
    #soundfile = al.Sndfile(target, 'w', al.Format('flac'), 1, new_rate)
    #soundfile.write_frames(signal)
    #soundfile.sync()

#Used in noise approximation algorithms
def normalize_noises(noisecsv, targetdir="noise8k/", new_rate=8000):
    noises = vad.readcsv(noisecsv)
    for noisefile in noises:
        resample_and_normalize_file(noisefile, targetdir+os.path.basename(noisefile), new_rate)

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
    return soundfile.read(path)
    """
    if al != None:
        sndfile = al.Sndfile(path, 'r')
        return sndfile.read_frames(sndfile.nframes), sndfile.samplerate
    else:
        try:
            print("Warning: no audiolab. Trying to read WAV: "+path)
            wav = wavfile.read(path)[1]
            wav = np.float64(wav)/np.iinfo(np.int16).max
            return wav
        except ValueError:
            return None
    """
