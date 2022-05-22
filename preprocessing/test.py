from lib.data_reader import NSynthDataset
from lib import specgrams_helper as spec
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('legend',fontsize= 12)
mpl.rc('axes', titlesize= 14)
import numpy as np
import scipy.io.wavfile as wavf

import tensorflow as tf
from magenta.models.gansynth.lib import specgrams_helper

source = os.getcwd()
path = '../data/nsynth-test.tfrecord'
path = os.path.join(source, path)

nsynth = NSynthDataset(path)
dataset = nsynth.get_dataset()

specgramhelper = spec.SpecgramsHelper(audio_length=64000,
                                      spec_shape=513,
                                      window_length=1024,
                                      sample_rate=16000,
                                      mel_downscale=1,
                                      ifreq=True,
                                      discard_dc=False,
                                      )

print(specgramhelper._get_symmetric_nfft_nhop())
for sample in dataset.take(1):
    # Look at audio signal
    print(sample['instrument_str'], sample['pitch'], sample['note'])
    time = np.linspace(0, 4, 64000)
    plt.plot(time, sample['audio'])
    plt.savefig('test.png')
    plt.close()

    # Look at stft
    stft = specgramhelper.wave_to_stft(sample['audio'])
    print(np.shape(stft))

    f, ax = plt.subplots(figsize=(8, 6))
    time = np.linspace(0, 4, 513)
    freq = np.linspace(0, 8000, 513)
    ax.pcolormesh(time, freq, np.log(np.abs(tf.transpose(stft).numpy())), cmap='viridis', shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    plt.savefig('stft_test.png')
    plt.close()

    # Look at specgrams
    specgram = specgramhelper.stft_to_specgram(stft)
    print(np.shape(specgram))

    f, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(time, freq, np.log(np.abs(tf.transpose(specgram[:, :, 0]).numpy())), cmap='viridis', shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    plt.savefig('spec_test.png')
    plt.close()

    # Look at melspec
    melspec = specgramhelper.specgram_to_melspecgram(specgram)
    print(np.shape(melspec))

    f, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(time, freq, np.log(np.abs(tf.transpose(melspec[:, :, 0]).numpy())), cmap='viridis',
                  shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    plt.savefig('melspec_test.png')
    plt.close()

    # Look at inversion
    wave = specgramhelper.melspecgram_to_wave(melspec)
    time = np.linspace(0, 4, 64000)
    plt.plot(time, wave.numpy())
    plt.savefig('inversion_test.png')
    plt.close()
    # Look at difference of audio signals
    difference = sample['audio'].numpy() - wave.numpy()
    plt.plot(time, difference)
    plt.savefig('difference.png')
    # Look at audio files
    wavf.write('in.wav', 16000, sample['audio'].numpy())
    wavf.write('out.wav', 16000, wave.numpy())
