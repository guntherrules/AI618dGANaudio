from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils
import numpy as np
import tensorflow as tf

# FFT Specgram Shapes
# pylint:disable=g-complex-comprehension
SPECGRAM_REGISTRY = {
    (nfft, hop): shape for nfft, hop, shape in zip(
        [256, 256, 512, 512, 1024, 1024],
        [64, 128, 128, 256, 256, 512],
        [[129, 1001, 2], [129, 501, 2], [257, 501, 2],
         [257, 251, 2], [513, 251, 2], [513, 126, 2]])
}
# pylint:enable=g-complex-comprehension


class NSynthDataset(object):
  """Dataset object to help manage the TFRecord loading."""

  def __init__(self, tfrecord_path, is_training=True):
    self.is_training = is_training
    self.record_path = tfrecord_path

  def get_example(self, batch_size):
    """Get a single example from the tfrecord file.
    Args:
      batch_size: Int, minibatch size.
    Returns:
      tf.Example protobuf parsed from tfrecord.
    """
    features = {
        "note": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
        "note_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
        "instrument": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
        "instrument_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
        "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
        "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
        "sample_rate": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
        "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32, default_value=[0] * 64000),
        "qualities": tf.io.FixedLenFeature([10], dtype=tf.int64, default_value=[0] * 10),
        "qualities_str": tf.io.VarLenFeature(dtype=tf.string),
        "instrument_family": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
        "instrument_family_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
        "instrument_source": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
        "instrument_source_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
    }
    dataset = tf.data.TFRecordDataset([self.record_path])
    serialized_example = next(dataset.as_numpy_iterator())
    example = tf.io.parse_single_example(serialized_example, features)
    return example

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
      crop = tf.image.random_crop(wav, [length])
      crop = tf.reshape(crop, [1, length])
    else:
      # fixed center crop
      offset = (64000 - length) // 2  # 24320
      crop = tf.slice(wav, [offset], [length])
      crop = tf.reshape(crop, [1, length])

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

      if hparams.pad:
        # Pad and crop specgram to 256x256
        num_padding = 2**int(np.ceil(np.log(shape[2]) / np.log(2))) - shape[2]
        tf.logging.info("num_pading: %d" % num_padding)
        specgram = tf.reshape(specgram, shape)
        specgram = tf.pad(specgram, [[0, 0], [0, 0], [0, num_padding], [0, 0]])
        specgram = tf.slice(specgram, [0, 0, 0, 0], [-1, shape[1] - 1, -1, -1])

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
