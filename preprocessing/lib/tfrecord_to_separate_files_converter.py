import pandas as pd
import numpy as np

from lib.data_reader import NSynthDataset
import os
import tensorflow as tf

class TfrecordConverter(object):
    """Helper functions convert tensorflow dataset to feather files"""

    def __init__(self, raw_dataset_path, save_path):
        self.save_path = save_path
        self.num_samples = self.get_num_samples(raw_dataset_path)
        nsynth = NSynthDataset(raw_dataset_path)
        self.dataset = nsynth.get_dataset()
        self.list_features = ['audio', 'qualities', 'qualities_str']
        self.txt_features = ['note_str', 'instrument_str', 'instrument_family_str', 'instrument_source_str']
        self.int_features = ['note', 'instrument', 'pitch', 'velocity', 'sample_rate', 'instrument_family', 'instrument_source']

    def tfrecord_to_feather(self, sample):
        """Convert tfrecord samples to dataframes"""
        sample['qualities'] = np.array(sample['qualities'])
        sample['qualities_str'] = np.array(tf.sparse.to_dense(sample['qualities_str']))
        sample['audio'] = np.array(sample['audio'])
        for list_feature in self.list_features:
            sample[list_feature] = [sample[list_feature]]
        for int_feature in self.int_features:
            sample[int_feature] = sample[int_feature].numpy()[0]
        for txt_feature in self.txt_features:
            sample[txt_feature] = bytes.decode(sample[txt_feature].numpy())
        df = pd.DataFrame.from_dict(sample)
        qual_str = ''.join(str(x) for x in df['qualities'][0])
        filename = os.path.join(self.save_path, df['note_str'][0] + '_' + qual_str + '.feather')
        df.to_feather(filename)

    def convert_dataset(self):
        n = 0.0
        for sample in self.dataset:
            self.tfrecord_to_feather(sample)
            n += 1.0
            print(n / self.num_samples)

    def get_num_samples(self, raw_data_path):
        if raw_data_path.__contains__('test'):
            return 4096.0
        elif raw_data_path.__contains__('valid'):
            return 12678.0
        elif raw_data_path.__contains__('train'):
            return 289205.0
