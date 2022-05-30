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
path = '../data/nsynth-test.tfrecord'
path = os.path.join(source, path)
output_path = os.path.join(source, 'test_output')

nsynth = NSynthDataset(path)
dataset = nsynth.get_dataset()
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
magmax, magmin, pmax, pmin = -100, 100, -100, 100
print(len(files))
random.seed(47)
files_subset = random.sample(files, 50)

for file in files_subset:
    # Look at audio signal
    sample = pd.read_feather(file)
    name = sample['note_str'][0]
    audio = sample['audio'][0]
    print(name)
    time = np.linspace(0, 4, 64000)
    plt.plot(time, audio)
    filename = os.path.join(output_path, name + '_wave.png')
    plt.savefig(filename)
    plt.close()

    # Look at stft
    print(specgramhelper._get_symmetric_nfft_nhop)
    stft = specgramhelper.wave_to_stft(audio)

    f, ax = plt.subplots(figsize=(8, 6))
    time = np.linspace(0, 4, spec_shape)
    freq = np.linspace(0, 8000, spec_shape)
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
    print(np.max(melspec[:, :, 0]), np.min(melspec[:, :, 0]))
    print(np.max(melspec[:, :, 1]), np.min(melspec[:, :, 1]), '\n')
    if np.max(melspec[:,:,0])>magmax:
        magmax = np.max(melspec[:,:,0])
    if np.max(melspec[:,:,1])>pmax:
        pmax = np.max(melspec[:,:,1])
    if np.min(melspec[:,:,0])<magmin:
        magmin = np.min(melspec[:,:,0])
    if np.min(melspec[:,:,1])<pmin:
        pmin = np.min(melspec[:,:,1])
    f, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(time, freq, tf.transpose(melspec[:, :, 0]).numpy(), cmap='viridis',
                  shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    filename = os.path.join(output_path, name + '_melspec.png')
    plt.colorbar(mesh)
    plt.savefig(filename)
    plt.close()

    # Look at normalized melspec
    normalized_melspec = specgramhelper.melspecgram_to_normalized_melspecgram(melspec)
    print(np.max(normalized_melspec))
    f, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(time, freq, tf.transpose(normalized_melspec[:, :, 0]).numpy(), cmap='viridis',
                         shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    filename = os.path.join(output_path, name + '_norm_melspec.png')
    plt.colorbar(mesh)
    plt.savefig(filename)
    plt.close()

    f, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(time, freq, tf.transpose(normalized_melspec[:, :, 1].numpy()), cmap='viridis',
                         shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    filename = os.path.join(output_path, name + '_norm_melspec_ph.png')
    plt.colorbar(mesh)
    plt.savefig(filename)
    plt.close()

    # Look at inversion
    wave = specgramhelper.normalized_melspecgram_to_wave(normalized_melspec)
    #wave = specgramhelper.melspecgram_to_wave(melspec)
    time = np.linspace(0, 4, 64000)
    plt.plot(time, wave.numpy())
    filename = os.path.join(output_path, name + '_inversion.png')
    plt.savefig(filename)
    plt.close()

    # Look at difference of audio signals
    difference = audio - wave.numpy()
    plt.plot(time, difference)
    filename = os.path.join(output_path, name + '_difference.png')
    plt.savefig(filename)

    # Look at audio files
    filename = os.path.join(output_path, name + '_in.wav')
    wavf.write(filename, 16000, audio)
    filename = os.path.join(output_path, name + '_out.wav')
    wavf.write(filename, 16000, wave.numpy())

print(magmax, magmin)
print(pmax, pmin)
