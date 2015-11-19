import sys
import os.path as path
import scipy.io as sio
import numpy as np
from pydub import AudioSegment

def run(sample_rate, in_file, out_file):
    sample_rate = int(sample_rate)
    basename = path.splitext(path.basename(in_file))[0]
    format = path.splitext(path.basename(out_file))[1][1:]
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
    sound = AudioSegment(data, metadata={
        'channels': 1,
        'sample_width': 2,
        'frame_rate': sample_rate,
        'frame_width': 2
        })
    sound.export(out_file, format=format)
    return sound, data

if __name__ == "__main__":
    run(*sys.argv[1:])
