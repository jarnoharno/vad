import numpy as np

# length should not be smaller than the end timestamp of the last segment
def convert(length, path):
    seq = np.loadtxt(path)[:,0:2].flatten().tolist()
    seq.append(length)
    return seq

if __name__ == "__main__":
    import sys
    print(sys.argv)
    seq = convert(float(sys.argv[1]),sys.argv[2])
    for t in seq:
        print(t)
