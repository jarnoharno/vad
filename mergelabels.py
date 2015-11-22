import numpy as np
import itertools as it

def mergefiles(files):
    # load flattened sequences
    lists = [np.loadtxt(file)[:,0:2].flatten().tolist() for file in files]
    return mergelists(lists)

# calling with sublists having an odd number of elements results in undefined
# behaviour
def mergelists(lists):
    # concatenate lists
    x = [y for l in lists for y in l]
    # rearrange into tuples marking beginning and end of an event
    x = sorted(it.izip(x,it.cycle([1,-1])))
    # merge by removing redundant elements
    events = 0
    seq = []
    for p in x:
        if (p[1] == 1 and events == 0) or (p[1] == -1 and events == 1):
            seq.append(p[0])
        events = events + p[1]
    return seq

if __name__ == "__main__":
    import sys
    seq = mergefiles(sys.argv[1:])
    for t in seq:
        print(t)
