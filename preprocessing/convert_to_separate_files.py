"""Protocol to split tfrecord dataset into single files for subset selection"""

from lib import tfrecord_to_separate_files_converter
import os

source = os.getcwd()
path = '../data/nsynth-train.tfrecord'
path = os.path.join(source, path)
save_path = os.path.join(source, '../data/melspecs/train')

converter = tfrecord_to_separate_files_converter.TfrecordConverter(raw_dataset_path=path, save_path=save_path)
converter.convert_dataset()
