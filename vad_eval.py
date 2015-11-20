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

def main(signalcsv, noisecsv, snrcsv, algorithmscsv, samplerate, tmppath, resultcsv):
    """
    combine signals and noises with all SNRs and write audio output to tmppath
    with given samplerate
    labels from signalcsv
    write to tmppath/signalname_noisename_+-SNR.flac
    call algorithm from algorithm.csv
    which writes to predictiondir/algoname_signalname_noisename_+-SNR.txt
    """
    signal_list = readcsv(signalcsv, True)
    noise_list = readcsv(noisecsv)
    snrlist = readcsv(snrcsv)
    try:
        snrlist = [float(x) for x in snrlist]
    except ValueError:
        print("Failed reading SNR definitions as float-values")
    algorithms = readcsv(algorithmcsv, True)
    combine(signal_list, noise_list, snrlist)

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
            noisy_signal = signal*snrdb2ratio(signal)+noise
            new_name = signal_file+"_"+noise_file
            soundfile = al.Sndfile(new_name, 'r', 'flac', 1, target_rate)
            soundfile.write_frames(noisy_signal)
            soundfile.sync()
            print("NOISE RMS:", noise_rms)

def noise_tilify(noise, rate, sfade):
    l = len(noise)/2
    head = noise[:l]
    tail = noise[l+1:]
    fade = min(l, rate*sfade)
    outgain = np.sin(math.pi*np.arange(0,1,1.0/fade))
    ingain = np.cos(math.pi*np.arange(0,1,1.0/fade))
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

def read_soundfile(filename):
        soundfile = al.Sndfile(filename, 'r')
        signal = soundfile.read_frames(soundfile.nframes)
        if soundfile.channels == 1:
            return signal, soundfile.samplerate
        else:
            return signal[:,0], soundfile.samplerate

""" Read vad csv-files """
def readcsv(filename, has_lists=False, delimiter=" "):
    res = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            if has_lists:
                res.append(row[0])
            else:
                res.append(row)
    return res

def predict(comb_files, labels, algos):
    #["matlab", "g279.m" "combined_list.txt", prediction_dir]
    for alg in algos:
        call(alg)

def evaluate(predicted_path, truthpath, resultpath):
    """ """

def g279(combined_files, prediction_dir):
    """" call matlab g279.m """
    """ store labels to csvfile"""
    pass

def txt2list(path):
    pass


#if __name__ == "__main__":
#    if len(argv == n):
#        main(argv*)
