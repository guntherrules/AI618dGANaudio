'''Script to get the global minimum and maximum of the melspec range for usage in normalizing the data'''

from lib import specgrams_helper as spec
import glob
import pandas as pd
import numpy as np

def get_value_range():
    """Get min and max values for magnitude and phase of melspectograms"""
    spec_shape = 256
    specgramhelper = spec.SpecgramsHelper(audio_length=64000,
                                          spec_shape=spec_shape,
                                          window_length=spec_shape*2,
                                          sample_rate=16000,
                                          mel_downscale=1,
                                          ifreq=True,
                                          )
    path = '../data/train/'
    files = glob.glob(path + '*.feather')
    print(len(files))

    magmax, magmin, pmax, pmin = -100, 100, -100, 100

    for file in files:
        sample = pd.read_feather(file)
        audio = sample['audio'][0]

        melspec = specgramhelper.wave_to_melspecgram(audio)
        if np.max(melspec[:, :, 0]) > magmax:
            magmax = np.max(melspec[:, :, 0])
        if np.max(melspec[:, :, 1]) > pmax:
            pmax = np.max(melspec[:, :, 1])
        if np.min(melspec[:, :, 0]) < magmin:
            magmin = np.min(melspec[:, :, 0])
        if np.min(melspec[:, :, 1]) < pmin:
            pmin = np.min(melspec[:, :, 1])

    return (magmax, magmin, pmax, pmin)

print(get_value_range())
