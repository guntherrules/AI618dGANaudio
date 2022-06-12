"""Protocol to split tfrecord dataset into single files for subset selection"""

from lib import tfrecord_to_separate_files_converter
import os

source = os.getcwd()
save_path = os.path.join(source, '../data/all_samples')
partitions = ['valid', 'test', 'train']

for partition in partitions:
    path = '../data/nsynth-{}.tfrecord'.format(partition)  # path from current dir to tfrecord-files
    load_path = os.path.join(source, path)
    converter = tfrecord_to_separate_files_converter.TfrecordConverter(raw_dataset_path=load_path, save_path=save_path)
    converter.convert_dataset()
