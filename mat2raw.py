import sys
import os.path as path
import scipy.io as sio
import numpy as np
from pydub import AudioSegment

def run(in_file, out_file):
    basename = path.splitext(path.basename(in_file))[0]
    data = sio.loadmat(in_file)[basename]
    if data.dtype == np.dtype('<i2'):
        pass
    elif data.dtype == np.dtype('>i2'):
        data = data.astype(np.dtype('<i2'))
    elif data.dtype == np.dtype('float64'):
        data = (data*np.iinfo(np.int16).max).astype(np.dtype('<i2'))
    else:
        print 'unsupported data type'
        sys.exit(1)
    print(data.flags)
    data = data.copy(order='C')
    data.tofile(out_file)
    return data

if __name__ == "__main__":
    run(*sys.argv[1:])
