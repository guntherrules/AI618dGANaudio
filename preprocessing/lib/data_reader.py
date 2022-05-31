import tensorflow as tf
import glob
from torch.utils.data import Dataset
from lib import specgrams_helper as spec
import pandas as pd
import os

class NSynthDataset(object):
    """Object to help manage the TFRecord loading."""

    def __init__(self, tfrecord_path):
        self.record_path = tfrecord_path
        self.features = {
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

    def parse(self, example):
        return tf.io.parse_single_example(example, self.features)

    def get_dataset(self):
        raw_dataset = tf.data.TFRecordDataset([self.record_path])
        return raw_dataset.map(self.parse)

class NSynthDatasetTorch(Dataset):
    """Pytorch dataset object for loading nsynth dataset or subset"""

    def __init__(self, select, path, label, spec_shape=256):
        self.paths = self.get_selection(select, path)
        self.spec_shape = spec_shape
        self.label = label
        self.specgrams_helper = spec.SpecgramsHelper(audio_length=64000,
                                                  spec_shape=spec_shape,
                                                  window_length=spec_shape*2,
                                                  sample_rate=16000,
                                                  mel_downscale=1,
                                                  ifreq=False,
                                                  )
        #self.transform = transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sample = pd.read_feather(path)
        audio = sample['audio'][0]
        melspec = self.specgrams_helper.wave_to_normalized_melspecgram(audio).numpy()
        #print(type(melspec.numpy()))
        if self.label == 'instrument_family':
            label = os.path.basename(path).split('_')[0]
        elif self.label == 'pitch':
            label = path.split('_')[2].split('-')[0]
        return melspec, label

    def get_selection(self, select, path):
        """Select a subset of filenames from nsynth dataset

        Args:
            select: dictionary of subset properties like pitchrange, instrument_name etc
            path: path to the audio files

        Returns:
            list or filenames in the subset
        """
        search_str = path
        for key in select.keys():
            if select[key] == None:
                search_str += '*'
            else:
                search_str += select[key]
            if key in ['instrument_name', 'pitch_range']:
                search_str += '-'
            elif key != 'qualities':
                search_str += '_'
            else:
                search_str += '.feather'
        return glob.glob(search_str)
