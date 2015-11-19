
def main(signalcsv, noisecsv, snrcsv, algorithmscsv, samplerate, tmppath, resultcsv):
    """ combine signals and noises with all SNRs and write audio output to tmppath
        with given samplerate
        labels from signalcsv
        write to tmppath/signalname_noisename_+-SNR.flac
        call algorithm from algorithm.csv
        which writes to predictiondir/algoname_signalname_noisename_+-SNR.txt
        """

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

def combine(signal, noise, snr):
    pass

if __name__ == "__main__":
    if len(argv == n):
        main(argv*)
