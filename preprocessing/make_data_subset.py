import numpy as np

from lib.dataset import get_selection
import lib.dataset as dataset

select = {'instrument_family_str': None,
          'instrument_source_str': 'acoustic',
          'instrument_name': None,
          'pitch_range': '0[2-8]4',
          'velocity': None,
          'qualities': None}
load_path = '../data/train'
save_path = '../data/train_subset'
print(get_selection(select, load_path))

torchDataset = dataset.RawDataset(select, load_path, resolution=8)
for idx in range(1):#range(torchDataset.__len__()):
    sample = torchDataset.__getitem__(idx)
    filename = sample[1].replace(load_path, save_path)
    specgram = sample[0].numpy()
    np.save(filename, specgram)
    #print(type(specgram), np.shape(specgram))


