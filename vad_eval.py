import os
import csv
from subprocess import call
from matplotlib import pyplot as plt
import librosa
import numpy as np
import math
import speech_processing as speech
import sys
import subprocess
import mergelabels
import multiprocessing
from tempfile import NamedTemporaryFile
from time import time
from simple_evaluation import compare_labels
import shutil
from sys import argv

try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

def preprocess(signalcsv="signals.txt", noisecsv="noise.txt", snrcsv="snr.txt", algocsv="algo.txt", tmppath="tmp/", resultpath="res/"):
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
    speech.combine(signal_list, noise_list, snrlist, tmppath)
    #call_vads(algorithms, tmppath, resultpath)
    #calculate_metrics(resultpath)

def resample_data(signalcsv="signals.txt", target="signal8k/", rate=8000):
    signal_list = readcsv(signalcsv)
    for signal in signal_list:
        speech.resample_and_normalize_file(signal, target+os.path.basename(signal), rate)

def main(signalcsv="signals.txt", noisecsv="noise.txt", snrcsv="snr.txt", algocsv="algo.txt", tmppath="tmp/", resultpath="res/", metricspath="eval/", clean_temps=False):
    signal_list = readcsv(signalcsv, True)
    noise_list = readcsv(noisecsv)
    snrlist = readcsv(snrcsv)
    try:
        snrlist = [float(x) for x in snrlist]
    except ValueError:
        print("Failed reading SNR definitions as float-values")
    pybin = "../bin/python"
    algolist = [
        #[pybin, "sos-vad.py"],
        [pybin, "ad-ltsd.py", "batch"],
        [pybin, "ltacs-vad.py", "batch"],
        [pybin, "rse-vad.py", "batch"] ,
        [pybin, "snac-vad.py", "batch"]
    ]
    print(signal_list)
    for signal in signal_list:
        speech.combine([signal], noise_list, snrlist, tmppath)
        sig_fn = os.path.basename(signal[0])
        print(signal, tmppath)
        shutil.copyfile("signal8k/"+sig_fn, tmppath+os.path.splitext(sig_fn)[0]+"_clean_40.0.flac")
        call_vads(algolist, tmppath, resultpath)
        for f in os.listdir(tmppath):
	    os.remove(tmppath+f)
    evaluation(resultpath, metricspath)

def evaluation(resultpath="res/", metricspath="eval/"):
    tr = load_truths()
    compute_all_metrics(tr, resultpath, metricspath)

def evaluate_for_speaker_distances(resultpath="res/", metricspath="eval/"):
    truths
    for ids in predictions():
        pass

def predictions(predictionpath="res/"):
     for fn in os.listdir(predictionpath):
        if os.path.splitext(fn)[1] == ".txt":
            yield(os.path.splitext(fn)[0], os.path.splitext(fn[0].split("_")))

def compute_all_metrics(truths, resultpath, metricspath):
    tasks = []
    for fn in os.listdir(resultpath):
        split1 = os.path.splitext(fn)
        if split1[1]==".txt":
            identifier = split1[0]
            ids = identifier.split("_")
            print(ids)
            if ids[1][0] in truths:
                #TODO: Only evaluate against combined labels for now
                tasks.append((identifier, truths[ids[1][0]]["combined"], resultpath+fn, metricspath))
    #pool = multiprocessing.Pool(None)
    #r = pool.map_async(compute_metrics, tasks)
    #r.wait()
    for task in tasks:
        compute_metrics(task)

def compute_metrics(args):
    id, truths, predictions, metricspath = args
    truths = segments_to_indexes(truths, "list")
    tr_fname = ""
    with open(metricspath+"/"+id+"_metrics.txt", "w") as metricf:
    #with NamedTemporaryFile('w') as tr, open(metricspath+"/"+id+"_metrics.txt", "w") as metricf:
        #np.savetxt(tr, truths, delimiter="\n")
        #tr.writelines([str(index[0])+'\n' for index in truths])
        #tr.flush()
        #tr_fname = tr.name
        #nullf = open(os.devnull, 'w')
        t = time()
        prediction_data = np.loadtxt(predictions)
        if len(prediction_data) > 1:
            tp, tn, fp, fn = compare_labels(np.loadtxt(predictions), truths)
            metricf.write(id+" TP "+str(tp)+"\n")
            metricf.write(id+" TN "+str(tn)+"\n")
            metricf.write(id+" FP "+str(fp)+"\n")
            metricf.write(id+" FN "+str(fn)+"\n")
        else:
            metricf.write(id+" FN 999 failed to find any\n")
        metricf.close()
        #call(["julia", "metric/metric.jl", id, tr_fname, predictions], stdout=metricf, stderr=nullf)
        #tr.close

def print_truths(truths):
    t = mergelabels.mergelists(truths)
    for l in t:
        print(l)

def read_metrics(metricfn):
    data = readcsv(metricfn, True)
    if len(data) > 0:
        m = {}
        id = data[0][0]
        info = id.split("_")
        {"algo":info[0], "signal":info[1], "noise":info[2],
            "snr":float(info[3])}
        for row in data:
            m[row[1]] = float(row[2])
        #if not "FP" in m:
        #    m["FP"] = m["I"]+m["M"]+m["OS"]+m["OE"]
        #if not "FN" in m:
        #    m["FN"] = m["D"]+m["F"]+m["US"]+m["UE"]
        try:
            acc = (m["TP"]+m["TN"])/(m["TP"]+m["TN"]+m["FP"]+m["FN"])
            recall = m["TP"]/(m["TP"]+m["FN"])
            precision = m["TP"]/(m["TP"]+m["FP"])
            fpr = m["FP"]/(m["TP"]+m["FN"])
            return id, {"algo":info[0], "signal":info[1], "noise":info[2],
                "snr":float(info[3]), "metrics":m, "ACC":acc, "TPR":recall, "PPV":precision}
        except KeyError:
            return None
    return None

def summarize(metricspath="eval/", group=None):
    metrics = {}
    for fn in os.listdir(metricspath):
        if os.path.splitext(fn)[1] == ".txt":
            res = read_metrics(metricspath+fn)
            if res != None:
                id, m = res
                metrics[id] = m
    perf = {}
    for k, m in metrics.iteritems():
        snr = float(k.split("_")[3])
        if snr not in perf:
            perf[snr] = {}
        if m['algo'] not in perf[snr]:
            perf[snr][m['algo']] = {'PPV':float(m['PPV']),'ACC':float(m['ACC']), 'TPR':float(m['TPR']), 'count':1}
        else:
            perf[snr][m['algo']]['ACC'] += float(m['ACC'])
            perf[snr][m['algo']]['PPV'] += float(m['PPV'])
            perf[snr][m['algo']]['TPR'] += float(m['TPR'])
            perf[snr][m['algo']]['count'] += 1
    for snr in perf.keys():
        for k,p in perf[snr].iteritems():
            p['ACC'] = p['ACC']/p['count']
            p['PPV'] = p['PPV']/p['count']
            p['TPR'] = p['TPR']/p['count']
    return metrics, perf

def plot_accuracy_snr(metrics):
    for key in metrics.keys:
        print(key)

def call_vads(algorithms, audiopath="tmp/", resultpath="res/"):
    #algorithms = readcsv(algocsv, True)
    for instruction in algorithms:
        print(instruction+[audiopath]+[resultpath])
        subprocess.check_call(instruction+[audiopath]+[resultpath])

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
            if has_lists == False:
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

def read_segment_file(path):
    with open(path) as sf:
        segments = []
        for l in sf.readlines():
            l = l.split('\t')
            segments.append([float(l[0]), float(l[1])])
        return segments
    print("Failed reading"+path)
    return None

def load_truths(paths=None):
    """ Returns combined truth values in a dictionary
    For example:
    {"s":[[voice-starttime, endtime],...],
     "g":[[voice-starttime, endtime],...]}"""
    if paths == None:
        paths = ['data/si.txt', 'data/sj.txt', 'data/gi.txt', 'data/gj.txt']
    data = {}
    for fn in paths:
        segments = read_segment_file(fn)
        identity = os.path.splitext(os.path.basename(fn))[0]
        scenario = identity[0]
        if scenario not in data: data[scenario] = {}
        data[scenario][identity] = segments
    for scene_id, scenario in data.iteritems():
        combined = []
        for segments in scenario.values():
            combined = combined+segments
        combined = sorted(combined)
        scenario['combined'] = combined
    return data

def segments_to_indexes(segments, style="tuples"):
    indexes = []
    for s in segments:
        indexes.append([s[0], "start"])
        indexes.append([s[1], "end"])
    indexes = sorted(indexes)
    last = "end"
    filtered = []
    for i in indexes:
        if i[1] != last:
            if style == "tuples":
                filtered.append(i)
            else:
                filtered.append(i[0])
        last = i[1]
    return filtered

def plot_segments(segments, cl='ti', ax=None):
    if ax == None:
        p = plt
    else:
        p = ax
    args = {}
    args['p'] = {'facecolor':'orange', 'alpha':0.25, 'linestyle':'dashed'}
    args['ti'] = {'facecolor':'blue', 'alpha':0.25}
    args['tj'] = {'facecolor':'green', 'alpha':0.25}
    a = args[cl]
    for segment in segments:
        p.axvspan(segment[0], segment[1], **a)

#create linear values for x-axis
def xspace(samples, frame_len, samplerate):
    return np.linspace(0,samples*frame_len/samplerate, samples)

if __name__ == "__main__":
    if len(argv) == 1:
        main()
    elif len(argv) == 4:
        main(tmppath=argv[1], resultpath=argv[2], metricspath=argv[3])
    else:
        print("usage: "+argv[1]+" [tmppath/ resultpath/ metricspath/]")
        print("  paths for temporary audiofiles, vad results and metrics")
