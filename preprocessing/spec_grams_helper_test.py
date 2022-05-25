from lib.data_reader import NSynthDataset
from lib import specgrams_helper as spec
import os
import numpy as np
import scipy.io.wavfile as wavf
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('legend', fontsize=12)
mpl.rc('axes', titlesize=14)

source = os.getcwd()
path = '../data/nsynth-test.tfrecord'
path = os.path.join(source, path)
output_path = os.path.join(source, 'test_output')

nsynth = NSynthDataset(path)
dataset = nsynth.get_dataset()

specgramhelper = spec.SpecgramsHelper(audio_length=64000,
                                      spec_shape=512,
                                      window_length=1024,
                                      sample_rate=16000,
                                      mel_downscale=1,
                                      ifreq=False,
                                      discard_dc=False,
                                      )

for sample in dataset.take(50):

    # Look at audio signal
    name = bytes.decode(sample['note_str'].numpy())
    time = np.linspace(0, 4, 64000)
    plt.plot(time, sample['audio'])
    filename = os.path.join(output_path, name + '_wave.png')
    plt.savefig(filename)
    plt.close()

    # Look at stft
    stft = specgramhelper.wave_to_stft(sample['audio'])

    f, ax = plt.subplots(figsize=(8, 6))
    time = np.linspace(0, 4, 513)
    freq = np.linspace(0, 8000, 513)
    ax.pcolormesh(time, freq, np.log(np.abs(tf.transpose(stft).numpy())), cmap='viridis', shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    filename = os.path.join(output_path, name + '_stft.png')
    plt.savefig(filename)
    plt.close()

    # Look at specgrams
    specgram = specgramhelper.stft_to_specgram(stft)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(time, freq, np.log(np.abs(tf.transpose(specgram[:, :, 0]).numpy())), cmap='viridis', shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    filename = os.path.join(output_path, name + '_spec.png')
    plt.savefig(filename)
    plt.close()

    # Look at melspec
    melspec = specgramhelper.specgram_to_melspecgram(specgram)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(time, freq, np.log(np.abs(tf.transpose(melspec[:, :, 0]).numpy())), cmap='viridis',
                  shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    filename = os.path.join(output_path, name + '_melspec.png')
    plt.savefig(filename)
    plt.close()

    # Look at inversion
    wave = specgramhelper.melspecgram_to_wave(melspec)
    time = np.linspace(0, 4, 64000)
    plt.plot(time, wave.numpy())
    filename = os.path.join(output_path, name + '_inversion.png')
    plt.savefig(filename)
    plt.close()

    # Look at difference of audio signals
    difference = sample['audio'].numpy() - wave.numpy()
    plt.plot(time, difference)
    filename = os.path.join(output_path, name + '_difference.png')
    plt.savefig(filename)

    # Look at audio files
    filename = os.path.join(output_path, name + '_in.wav')
    wavf.write(filename, 16000, sample['audio'].numpy())
    filename = os.path.join(output_path, name + '_out.wav')
    wavf.write(filename, 16000, wave.numpy())
