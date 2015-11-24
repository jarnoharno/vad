import itertools as it

def readseq(path):
    with open(path, 'r') as f:
        return [float(line) for line in f]

def mergefiles(files):
    lists = [readseq(f) for f in files]
    return mergelists(lists)

# calling with sublists having and even number of elements is undefined
def mergelists(lists):
    # get the longest sequence
    t = max([l[-1] for l in lists])
    # prune last timestamps
    lists = [l[0:-1] for l in lists]
    # concatenate lists
    x = [y for l in lists for y in l]
    # append last timestamp
    x.append(t)
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
