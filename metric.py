import itertools as it

def readseq(path):
    with open(path, 'r') as f:
        return [float(line) for line in f]

def readtup(path,index):
    seq = readseq(path)
    return list(it.izip(seq,it.repeat(index)))

def print_evaluation_metrics(actual_seq_file, predicted_seq_file):
    events = sorted(readtup(actual_seq_file,0)+readtup(predicted_seq_file,1))
    # confusion matrix (actual on rows, predicted on columns
    confusion_matrix = [[0,0],[0,0]]
    # confusion matrix state
    s = [0,0]
    # last timestamp
    t = 0
    for i in events:
        confusion_matrix[s[0]][s[1]] += i[0]-t
        t = i[0]
        # flip state index
        s[i[1]] = not s[i[1]]
    return confusion_matrix
