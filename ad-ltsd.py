#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: ltsd.py
# $Date: Sun Jul 19 17:53:59 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

from sys import argv
import os
from scipy.io import wavfile
import matplotlib
#matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyssp.vad.ltsd import LTSD
from pyssp.vad.ltsd import AdaptiveLTSD

import soundfile
import librosa
import multiprocessing

MAGIC_NUMBER = 0.04644

class LTSD_VAD(object):
    ltsd = None
    order = 5

    fs = 0
    window_size = 0
    window = 0

    lambda0 = 0
    lambda1 = 0

    noise_signal = None

    def init_params_by_noise(self, fs, noise_signal):
        noise_signal = self._mononize_signal(noise_signal)
        self.noise_signal = np.array(noise_signal)
        self._init_window(fs)
        ltsd = AdaptiveLTSD(self.window_size, self.window, self.order)
        res, ltsds = ltsd.compute_with_noise(noise_signal,
                noise_signal)
        print(res, ltsds)
        max_ltsd = max(ltsds)
        self.lambda0 = max_ltsd * 1.1
        self.lambda1 = self.lambda0 * 2.0
        print 'max_ltsd =', max_ltsd
        print 'lambda0 =', self.lambda0
        print 'lambda1 =', self.lambda1

    def plot_ltsd(self, fs, signal):
        signal = self._mononize_signal(signal)
        res, ltsds = self._get_ltsd().compute_with_noise(signal, self.noise_signal)
        plt.plot(ltsds)
        plt.show()

    def filter(self, signal):
        signal = self._mononize_signal(signal)
        res, ltsds = self._get_ltsd().compute_with_noise(signal, self.noise_signal)
        voice_signals = []
        res = [(start * self.window_size / 2, (finish + 1) * self.window_size
                / 2) for start, finish in res]
        print res, len(ltsds) * self.window_size / 2
        for start, finish in res:
            voice_signals.append(signal[start:finish])
        try:
            return np.concatenate(voice_signals), res
        except:
            return np.array([]), []

    def segments(self, signal):
        signal = self._mononize_signal(signal)
        res, ltsds = self._get_ltsd().compute_with_noise(signal, self.noise_signal)
        voice_signals = []
        res = [(start * self.window_size / 2, (finish + 1) * self.window_size
                / 2) for start, finish in res]
        return res, len(ltsds) * self.window_size / 2

    def _init_window(self, fs):
        self.fs = fs
        #self.window_size = int(MAGIC_NUMBER * fs)
        self.window_size = 320
        print(self.window_size)
        self.window = np.hanning(self.window_size)

    def _get_ltsd(self, fs=None):
        if fs is not None and fs != self.fs:
            self._init_window(fs)
        return AdaptiveLTSD(self.window_size, self.window, self.order,
                lambda0=self.lambda0, lambda1=self.lambda1)

    def _mononize_signal(self, signal):
        if signal.ndim > 1:
            signal = signal[:,0]
        return signal

def compute_vad(args):
    filename, path, resultpath = args
    signame = os.path.basename(os.path.splitext(filename)[0])
    ids = signame.split("_")
    print("computing: "+path+filename)
    bg_signal, rate = soundfile.read(path+filename)
    ltsd = LTSD_VAD()
    bg_signal=bg_signal[:2000]
    print(bg_signal)
    ltsd.init_params_by_noise(rate, bg_signal)
    signal, rate = soundfile.read(path+filename)
    #vaded_signal = ltsd.filter(signal)
    segments, sig_len = ltsd.segments(signal)
    #seconds = float(len(sig))/rate
    res_name = resultpath+"/ad-ltsd_"+os.path.basename(os.path.splitext(filename)[0])+".txt"
    segments = librosa.core.samples_to_time(segments, rate).tolist()
    len_s = librosa.core.samples_to_time(sig_len, rate)
    write_results(segments, res_name, len_s)

def write_results(segments, res_name, l):
    indexes = []
    for s in segments:
        indexes += s
    #indexes.append(l[0])
    print("writing "+res_name)
    f = open(res_name, 'w')
    f.write("\n".join([str(x) for x in indexes]))
    f.close()

def main():
    #fs, bg_signal = wavfile.read(sys.argv[1])
    if argv[1] == 'batch':
        files = []
        for f in os.listdir(argv[2]):
            if os.path.splitext(f)[1] == ".flac":
                files.append(f)
        args = [(f, argv[2], argv[3]) for f in files]
        pool = multiprocessing.Pool(10)
        r = pool.map_async(compute_vad, args)
        r.wait()
        pool.close()
        pool.join()
        #for a in args:
        #    compute_vad(a)
    else:
        bg_signal, fs = soundfile.read(argv[1])
        ltsd = LTSD_VAD()
        bg_signal=bg_signal[:2000]
        print(bg_signal)
        ltsd.init_params_by_noise(fs, bg_signal)
        signal, fs = soundfile.read(argv[1])
        #vaded_signal = ltsd.filter(signal)
        segments, sig_len = ltsd.segments(signal)
        print(ltsd.segments(signal)[0])
        #wavfile.write('vaded.wav', fs, vaded_signal)

if __name__ == '__main__':
    main()

# vim: foldmethod=marker
