import reader_original
import reader
import os
import pandas as pd
import utils

import tensorflow as tf
source = os.getcwd()
path = '../data/nsynth-test.tfrecord'
path = os.path.join(source, path)

nsynth = reader_original.NSynthDataset(path)
class HParams():
    def __init__(self,
                 batch_size,
                 n_fft = False,
                 hop_length = False,
                 mask = False,
                 log_mag = False,
                 re_im = False,
                 dphase = True,
                 mag_only = False,
                 pad = False,
                 raw_audio=False):
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mask = mask
        self.log_mag = log_mag
        self.re_im = re_im
        self.dphase = dphase
        self.mag_only = mag_only
        self.pad = pad
        self.raw_audio = raw_audio

hparams = HParams(1, 1024, 512)
print(nsynth.get_baseline_batch(hparams))
print(nsynth.get_baseline_batch(hparams)['spectrogram'])
spec = utils.form_image_grid(nsynth.get_baseline_batch(hparams)['spectrogram'], grid_shape=[1, 1], image_shape=[513, 126], num_channels=2)
utils.specgram_summaries(spec, name='test', hparams=hparams)