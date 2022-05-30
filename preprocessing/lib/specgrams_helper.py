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

"""Helper object for transforming audio to spectra.

Handles transformations between waveforms, stfts, spectrograms,
mel-spectrograms, and instantaneous frequency (specgram).
"""

import lib.spectral_ops as spectral_ops
import numpy as np
import tensorflow.compat.v1 as tf


class SpecgramsHelper(object):
  """Helper functions to compute specgrams."""

  def __init__(self, audio_length, spec_shape, window_length,
               sample_rate, mel_downscale, ifreq=True):
    self._audio_length = audio_length
    self._spec_shape = spec_shape
    self._window_length = window_length
    self._sample_rate = sample_rate
    self._mel_downscale = mel_downscale
    self._ifreq = ifreq
    self._nfft, self._nhop = self._get_symmetric_nfft_nhop
    self._eps = 1.0e-6
    # Normalization constants
    self._a, self._b = self.compute_normalization()

  def _safe_log(self, x):
    return tf.log(x + self._eps)

  def _get_symmetric_nfft_nhop(self):
    n_freq_bins = self._spec_shape
    # Power of two only has 1 nonzero in binary representation
    is_power_2 = bin(n_freq_bins).count('1') == 1
    if not is_power_2:
      raise ValueError('Wrong spec_shape. Number of frequency bins must be '
                       'a power of 2 plus 1, not %d' % n_freq_bins)
    nfft = n_freq_bins * 2
    nhop = int(2 * (self._audio_length - self._window_length) / nfft)
    return (nfft, nhop)

  def wave_to_stft(self, wave):
    """Convert from waves to complex stfts.

    Args:
      wave: Tensor of the waveform, shape [time].

    Returns:
      stft: Complex64 tensor of stft, shape [time, freq].
    """
    stft = tf.signal.stft(
        wave,
        frame_length=self._nfft,
        frame_step=self._nhop,
        fft_length=self._nfft,
        pad_end=False)[1:, 1:]
    stft_shape = stft.get_shape()

    if stft_shape[0] != self._spec_shape:
      raise ValueError(
          'Spectrogram returned the wrong shape {}, is not the same as the '
          'constructor spec_shape {}.'.format(stft_shape, self._spec_shape))
    return stft

  def stft_to_wave(self, stft):
    """Convert from complex stft to waves.

    Args:
      stft: Complex64 tensor of stft, shape [time, freq].

    Returns:
      wave: Tensor of the waveform, shape [time].
    """
    stft = tf.pad(stft, [[1, 0], [1, 0]])
    wave_resyn = tf.signal.inverse_stft(
        stfts=stft,
        frame_length=self._window_length,
        frame_step=self._nhop,
        fft_length=self._nfft,
        window_fn=tf.signal.inverse_stft_window_fn(
            frame_step=self._nhop))

    return wave_resyn

  def stft_to_specgram(self, stft):
    """Converts stft to specgram.

    Args:
      stft: Complex64 tensor of stft, shape [time, freq].

    Returns:
      specgram: Tensor of log magnitudes and instantaneous frequencies,
        shape [time, freq, 2].
    """

    logmag = self._safe_log(tf.abs(stft))

    phase_angle = tf.angle(stft)
    if self._ifreq:
      p = spectral_ops.instantaneous_frequency(phase_angle)
    else:
      p = phase_angle / np.pi

    return tf.concat(
        [logmag[:, :, tf.newaxis], p[:, :, tf.newaxis]], axis=-1)

  def specgram_to_stft(self, specgram):
    """Converts specgram to stft.

    Args:
      specgram: Tensor of log magnitudes and instantaneous frequencies,
        shape [time, freq, 2].

    Returns:
      stft: Complex64 tensor of stft, shape [time, freq].
    """
    logmag = specgram[:, :, 0]
    p = specgram[:, :, 1]

    mag = tf.exp(logmag)

    if self._ifreq:
      phase_angle = tf.cumsum(p * np.pi, axis=-2)
    else:
      phase_angle = p * np.pi

    return spectral_ops.polar2rect(mag, phase_angle)[:, :]

  def _linear_to_mel_matrix(self):
    """Get the mel transformation matrix."""
    num_freq_bins = self._nfft // 2
    lower_edge_hertz = 0.0
    upper_edge_hertz = self._sample_rate / 2.0
    num_mel_bins = num_freq_bins // self._mel_downscale
    return spectral_ops.linear_to_mel_weight_matrix(
        num_mel_bins, num_freq_bins, self._sample_rate, lower_edge_hertz,
        upper_edge_hertz)

  def _mel_to_linear_matrix(self):
    """Get the inverse mel transformation matrix."""
    m = self._linear_to_mel_matrix()
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

  def specgram_to_melspecgram(self, specgram):
    """Converts specgram to melspecgrams.

    Args:
      specgram: Tensor of log magnitudes and instantaneous frequencies,
        shape [time, freq, 2].

    Returns:
      melspecgram: Tensor of log magnitudes and instantaneous frequencies,
        shape [time, freq, 2], mel scaling of frequencies.
    """
    if self._mel_downscale is None:
      return specgram

    logmag = specgram[:, :, 0]
    p = specgram[:, :, 1]

    mag2 = tf.exp(2.0 * logmag)
    phase_angle = tf.cumsum(p * np.pi, axis=-2)

    l2mel = tf.to_float(self._linear_to_mel_matrix())
    logmelmag2 = self._safe_log(tf.tensordot(mag2, l2mel, 1))
    mel_phase_angle = tf.tensordot(phase_angle, l2mel, 1)
    mel_p = spectral_ops.instantaneous_frequency(mel_phase_angle)

    return tf.concat(
        [logmelmag2[:, :, tf.newaxis], mel_p[:, :, tf.newaxis]], axis=-1)

  def melspecgram_to_specgram(self, melspecgram):
    """Converts melspecgram to specgram.

    Args:
      melspecgram: Tensor of log magnitudes and instantaneous frequencies,
        shape [time, freq, 2], mel scaling of frequencies.

    Returns:
      specgram: Tensor of log magnitudes and instantaneous frequencies,
        shape [time, freq, 2].
    """
    if self._mel_downscale is None:
      return melspecgram

    logmelmag2 = melspecgram[:, :, 0]
    mel_p = melspecgram[:, :, 1]

    mel2l = tf.to_float(self._mel_to_linear_matrix())
    mag2 = tf.tensordot(tf.exp(logmelmag2), mel2l, 1)
    logmag = 0.5 * self._safe_log(mag2)
    mel_phase_angle = tf.cumsum(mel_p * np.pi, axis=-2)
    phase_angle = tf.tensordot(mel_phase_angle, mel2l, 1)
    p = spectral_ops.instantaneous_frequency(phase_angle)

    return tf.concat(
        [logmag[:, :, tf.newaxis], p[:, :, tf.newaxis]], axis=-1)

  def normalize_to_tanh(self, input_value_range):
    input_interval = float(input_value_range[1] - input_value_range[0])
    a = 2.0 / input_interval
    b = - 2.0 * input_value_range[0] / input_interval - 1.0
    return a, b

  def compute_normalization(self):
    mag_a, mag_b = self.normalize_to_tanh((-13.815511, 10.17237))
    p_a, p_b = self.normalize_to_tanh((-2.6498687, 2.6647818))
    return [mag_a, p_a], [mag_b, p_b]

  def melspecgram_to_normalized_melspecgram(self, melspecgram):
    """Normalize values to range of -1 to 1"""
    a = tf.constant(self._a, dtype=melspecgram.dtype)
    b = tf.constant(self._b, dtype=melspecgram.dtype)
    return tf.clip_by_value(a * melspecgram + b, -1.0, 1.0)

  def normalized_melspecgram_to_melspecgram(self, normalized_specgram):
    a = tf.constant(self._a, dtype=normalized_specgram.dtype)
    b = tf.constant(self._b, dtype=normalized_specgram.dtype)
    return (normalized_specgram - b)/a

  def stft_to_melspecgram(self, stft):
    """Converts stft to mel-spectrogram."""
    return self.specgram_to_melspecgram(self.stft_to_specgram(stft))

  def melspecgram_to_stft(self, melspecgram):
    """Converts mel-spectrogram to stft."""
    return self.specgram_to_stft(self.melspecgram_to_specgram(melspecgram))

  def wave_to_specgram(self, wave):
    """Converts wave to spectrogram."""
    return self.stft_to_specgram(self.wave_to_stft(wave))

  def specgram_to_wave(self, specgram):
    """Converts spectrogram to stft."""
    return self.stft_to_wave(self.specgram_to_stft(specgram))

  def wave_to_melspecgram(self, wave):
    """Converts wave to mel-spectrogram."""
    return self.stft_to_melspecgram(self.wave_to_stft(wave))

  def melspecgram_to_wave(self, melspecgram):
    """Converts mel-spectrogram to stft."""
    return self.stft_to_wave(self.melspecgram_to_stft(melspecgram))

  def wave_to_normalized_melspecgram(self, wave):
    """Converts wave to normalized melspecgram"""
    return self.melspecgram_to_normalized_melspecgram(self.wave_to_melspecgram(wave))

  def normalized_melspecgram_to_wave(self, normalized_melspecgram):
    """Converts normalized melspecgram to wave"""
    return self.melspecgram_to_wave(self.normalized_melspecgram_to_melspecgram(normalized_melspecgram))
