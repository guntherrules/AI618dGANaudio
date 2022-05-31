from lib import dataset
import numpy as np
from torch.utils.data import DataLoader

select = {'instrument_family_str': None,
          'instrument_source_str': 'acoustic',
          'instrument_name': None,
          'pitch_range': '0[2-8]4',
          'velocity': None,
          'qualities': None}
path = '../data/train/'

torchDataset = dataset.MultiResolutionDataset(select, path, resolution=8)
train_dataloader = DataLoader(torchDataset, batch_size=64, shuffle=True)
train_features = next(iter(train_dataloader))
print(train_features.size())

torchDataset = dataset.MultiLabelResolutionDataset(select, path, resolution=8)
train_dataloader = DataLoader(torchDataset, batch_size=64, shuffle=True)
train_features = next(iter(train_dataloader))
print(train_features[0].size(), train_features[1:])

torchDataset = dataset.MultiLabelAllDataset(select, path, resolution=8)
train_dataloader = DataLoader(torchDataset, batch_size=64, shuffle=True)
train_features = next(iter(train_dataloader))
print(train_features[0].size(), train_features[1:])
