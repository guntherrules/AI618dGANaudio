from lib import data_reader
from torch.utils.data import DataLoader

select = {'instrument_family_str': None,
          'instrument_source_str': 'acoustic',
          'instrument_name': None,
          'pitch_range': '0[3-8]4',
          'velocity': None,
          'qualities': None}
path = '../data/train/'

torchDataset_pitch = data_reader.NSynthDatasetTorch(select, path, label='pitch')
torchDataset_family = data_reader.NSynthDatasetTorch(select, path, label='instrument_family')

train_dataloader = DataLoader(torchDataset_pitch, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(train_features, train_labels)

train_dataloader = DataLoader(torchDataset_family, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(train_features, train_labels)