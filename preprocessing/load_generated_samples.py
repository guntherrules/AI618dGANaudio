'''
Script to load npy-files generated with dGAN and convert them back to audio. Additionally plot the wave signal
and save the image as well.
'''

from lib.data_reader import NSynthDataset
from lib import specgrams_helper as spec
import os
import numpy as np
import scipy.io.wavfile as wavf
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import pandas as pd
import random

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('legend', fontsize=12)
mpl.rc('axes', titlesize=14)

source = os.getcwd()
# Put your path to generated npy files here
path = '../data/generated/sampleS/'
# Define a designated output path for the audio files
path = os.path.join(source, path)
output_path = path

# Put the shape of the generated tensors here
spec_shape = 256
specgramhelper = spec.SpecgramsHelper(audio_length=64000,
                                      spec_shape=spec_shape,
                                      window_length=spec_shape*2,
                                      sample_rate=16000,
                                      mel_downscale=1,
                                      ifreq=True,
                                      )
files = glob.glob(path + '*.npy')

for file in files:
    batch = np.load(file)
    for i in range(len(batch)):
        time = np.linspace(0, 4, spec_shape)
        freq = np.linspace(0, 8000, spec_shape)
        name = os.path.basename(file).replace('.npy', '')

        melspec = batch[i, :, :, :]
        melspec = np.transpose(melspec, (1, 2, 0))

        # Look at inversion
        wave = specgramhelper.normalized_melspecgram_to_wave(melspec)
        time = np.linspace(0, 4, 64000)
        plt.plot(time, wave.numpy())
        filename = os.path.join(output_path, name + '_audio' + str(i) + '.png')
        plt.savefig(filename)
        plt.close()

        # Look at audio files
        filename = os.path.join(output_path, name + '_audio' + str(i) + '.wav')
        wavf.write(filename, 16000, wave.numpy())

