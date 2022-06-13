'''Script to select a subset of nsynth data out of the separated dataset'''

import numpy as np

from lib.dataset import get_selection
import lib.dataset as dataset

select = {'instrument_family_str': None,
          'instrument_source_str': 'acoustic',
          'instrument_name': None,
          'pitch_range': '0[2-8]4',
          'velocity': None,
          'qualities': None}

load_path = '../data/all_samples'
save_path = '../data/all_samples_512'
size = len(get_selection(select, load_path, 'feather'))

torchDataset = dataset.RawDataset(select, load_path, resolution=512)

for idx in range(torchDataset.__len__()):
    sample = torchDataset.__getitem__(idx)
    filename = sample[1].replace(load_path, save_path)
    specgram = sample[0]
    np.save(filename, specgram)
    print(idx/size)
