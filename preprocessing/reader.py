# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module to load the Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils
import numpy as np
import os
import typing as t
import uuid

import tensorflow as tf
from pandas import DataFrame, Series
from pandas.api.types import is_integer_dtype, is_float_dtype, is_string_dtype

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


SPECGRAM_REGISTRY = {
    (nfft, hop): shape for nfft, hop, shape in zip(
        [256, 256, 512, 512, 1024, 1024],
        [64, 128, 128, 256, 256, 512],
        [[129, 1001, 2], [129, 501, 2], [257, 501, 2],
         [257, 251, 2], [513, 251, 2], [513, 126, 2]])
}


class NSynthDataset(object):
    """Dataset object to help manage the TFRecord loading."""

    def __init__(self, tfrecord_path):
        self.record_path = tfrecord_path

    def serialize_example(self):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        features = {
            "note": _int64_feature(1),
            "note_str": _bytes_feature(b"flute-50-50"),
            "instrument": _int64_feature(1),
            "instrument_str": _bytes_feature(b"flute-flute-50-50"),
            "pitch": _int64_feature(1),
            "velocity": _int64_feature(1),
            "sample_rate": _int64_feature(1),
            "audio": _int64_feature(1),
            "qualities": _int64_feature(1),
            "qualities_str": _bytes_feature(b"dark"),
            "instrument_family": _int64_feature(1),
            "instrument_family_str": _bytes_feature(b"bass"),
            "instrument_source": _int64_feature(1),
            "instrument_source_str": _bytes_feature(b"acoustic")
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def parse_tfr_element(element):
        features = {
            "note": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "note_str": tf.io.FixedLenFeature([], dtype=tf.string),
            "instrument": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "instrument_str": tf.io.FixedLenFeature([], dtype=tf.string),
            "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "sample_rate": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32),
            "qualities": tf.io.FixedLenFeature([10], dtype=tf.int64),
            "qualities_str": tf.io.FixedLenFeature([10], dtype=tf.string),
            "instrument_family": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "instrument_family_str": tf.io.FixedLenFeature([], dtype=tf.string),
            "instrument_source": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "instrument_source_str": tf.io.FixedLenFeature([], dtype=tf.string)
        }
        #example = Example(features=Features(feature=features))
        example = tf.io.parse_single_example(element, features)
        return example

    def get_dataset(self):
        # create the dataset
        dataset = tf.data.TFRecordDataset([self.record_path])
        # pass every single feature through our mapping function
        dataset = dataset.map(
            self.parse_tfr_element
        )

        return dataset

    def get_wavenet_batch(self, batch_size, length=64000):
        """Get the Tensor expressions from the reader.
        Args:
          batch_size: The integer batch size.
          length: Number of timesteps of a cropped sample to produce.
        Returns:
          A dict of key:tensor pairs. This includes "pitch", "wav", and "key".
        """
        example = self.get_example(batch_size)
        wav = example["audio"]
        wav = tf.slice(wav, [0], [64000])
        pitch = tf.squeeze(example["pitch"])
        key = tf.squeeze(example["note_str"])

        if self.is_training:
            # random crop
            crop = tf.random_crop(wav, [length])
            crop = tf.reshape(crop, [1, length])
            key, crop, pitch = tf.train.shuffle_batch(
                [key, crop, pitch],
                batch_size,
                num_threads=4,
                capacity=500 * batch_size,
                min_after_dequeue=200 * batch_size)
        else:
            # fixed center crop
            offset = (64000 - length) // 2  # 24320
            crop = tf.slice(wav, [offset], [length])
            crop = tf.reshape(crop, [1, length])
            key, crop, pitch = tf.train.shuffle_batch(
                [key, crop, pitch],
                batch_size,
                num_threads=4,
                capacity=500 * batch_size,
                min_after_dequeue=200 * batch_size)

        crop = tf.reshape(tf.cast(crop, tf.float32), [batch_size, length])
        pitch = tf.cast(pitch, tf.int32)
        return {"pitch": pitch, "wav": crop, "key": key}

    def get_baseline_batch(self, hparams):
        """Get the Tensor expressions from the reader.
        Args:
          hparams: Hyperparameters object with specgram parameters.
        Returns:
          A dict of key:tensor pairs. This includes "pitch", "wav", and "key".
        """
        example = self.get_example(hparams.batch_size)
        audio = tf.slice(example["audio"], [0], [64000])
        audio = tf.reshape(audio, [1, 64000])
        pitch = tf.slice(example["pitch"], [0], [1])
        velocity = tf.slice(example["velocity"], [0], [1])
        instrument_source = tf.slice(example["instrument_source"], [0], [1])
        instrument_family = tf.slice(example["instrument_family"], [0], [1])
        qualities = tf.slice(example["qualities"], [0], [10])
        qualities = tf.reshape(qualities, [1, 10])

        # Get Specgrams
        hop_length = hparams.hop_length
        n_fft = hparams.n_fft
        if hop_length and n_fft:
            specgram = utils.tf_specgram(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                mask=hparams.mask,
                log_mag=hparams.log_mag,
                re_im=hparams.re_im,
                dphase=hparams.dphase,
                mag_only=hparams.mag_only)
            shape = [1] + SPECGRAM_REGISTRY[(n_fft, hop_length)]
            if hparams.mag_only:
                shape[-1] = 1
            specgram = tf.reshape(specgram, shape)
            tf.logging.info("SPECGRAM BEFORE PADDING", specgram)

            if hparams.pad:
                # Pad and crop specgram to 256x256
                num_padding = 2**int(np.ceil(np.log(shape[2]) / np.log(2))) - shape[2]
                tf.logging.info("num_pading: %d" % num_padding)
                specgram = tf.reshape(specgram, shape)
                specgram = tf.pad(specgram, [[0, 0], [0, 0], [0, num_padding], [0, 0]])
                specgram = tf.slice(specgram, [0, 0, 0, 0], [-1, shape[1] - 1, -1, -1])
                tf.logging.info("SPECGRAM AFTER PADDING", specgram)

        # Form a Batch
        if self.is_training:
            (audio, velocity, pitch, specgram,
            instrument_source, instrument_family,
            qualities) = tf.train.shuffle_batch(
                [
                   audio, velocity, pitch, specgram,
                   instrument_source, instrument_family, qualities
                ],
                batch_size=hparams.batch_size,
                capacity=20 * hparams.batch_size,
                min_after_dequeue=10 * hparams.batch_size,
                enqueue_many=True)
        elif hparams.batch_size > 1:
            (audio, velocity, pitch, specgram,
            instrument_source, instrument_family, qualities) = tf.train.batch(
                [
                   audio, velocity, pitch, specgram,
                   instrument_source, instrument_family, qualities
                ],
                batch_size=hparams.batch_size,
                capacity=10 * hparams.batch_size,
                enqueue_many=True)

        audio.set_shape([hparams.batch_size, 64000])

        batch = dict(
            pitch=pitch,
            velocity=velocity,
            audio=audio,
            instrument_source=instrument_source,
            instrument_family=instrument_family,
            qualities=qualities,
            spectrogram=specgram)

        return batch

"""
Inspired from pandas-tfrecords:
https://github.com/schipiga/pandas-tfrecords/tree/master/pandas_tfrecords
https://pypi.org/project/pandas-tfrecords/
"""




def to_tfrecords(data: DataFrame,
               feature_columns: t.List[str],
               folder: str,
               file_name: str = None,
               compress: bool = False):

  tfrecords = get_tfrecords(data, feature_columns)

  file_name = write_tfrecords(tfrecords, folder, file_name, compress=compress)

  print(f'TFRecord saved to {file_name}')

def write_tfrecords(tfrecords: Example,
                  folder: str,
                  file_name: str = None,
                  compress: bool = False,
                  ) -> str:
  compression_ext = '.gz' if compress else ''

  os.makedirs(folder, exist_ok=True)

  uid = str(uuid.uuid4())
  opts = {}

  if compress:
      opts['options'] = tf.io.TFRecordOptions(
          compression_type='GZIP',
          compression_level=9,
      )

  if not file_name:
      file_name = f'part-{str(0).zfill(5)}-{uid}.tfrecord{compression_ext}'
  else:
      file_name = file_name + f'.tfrecord{compression_ext}'

  file_path = os.path.join(folder, file_name)

  with tf.io.TFRecordWriter(file_path, **opts) as writer:
      writer.write(tfrecords.SerializeToString())

  return file_name

def get_tfrecords(data: DataFrame, columns: t.List[str]) -> Example:

  for c in columns:
      if c not in data.columns:
          raise ValueError(f'Features missing from input data: {c}')

  samples = len(data)
  features = {column: get_feature(data[column]) for column in columns}
  features['samples'] = Feature(int64_list=Int64List(value=[samples]))

  return Example(features=Features(feature=features))

