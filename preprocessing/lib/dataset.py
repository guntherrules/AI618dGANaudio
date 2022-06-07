from io import BytesIO

# import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import random
import torch
import glob
import pandas as pd
import numpy as np
from lib import specgrams_helper as spec


def get_selection(select, path):
    """Select a subset of filenames from nsynth dataset

    Args:
        select: dictionary of subset properties like pitchrange, instrument_name etc
        path: path to the audio files

    Returns:
        list or filenames in the subset
    """
    search_str = path + '/'
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

class RawDataset(Dataset):
    """Pytorch dataset object for loading raw (i.e. audio format) nsynth dataset or subset"""

    def __init__(self, select, path, resolution=256):
        self.audiolist = get_selection(select, path)
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.RandomHorizontalFlip(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        self.specgrams_helper = spec.SpecgramsHelper(audio_length=64000,
                                                     spec_shape=resolution,
                                                     window_length=resolution * 2,
                                                     sample_rate=16000,
                                                     mel_downscale=1,
                                                     ifreq=True,
                                                     )

    def __len__(self):
        return len(self.audiolist)

    def __getitem__(self, idx):
        path = self.audiolist[idx]
        sample = pd.read_feather(path)
        audio = sample['audio'][0]
        melspec = self.specgrams_helper.wave_to_normalized_melspecgram(audio).numpy()
        melspec = self.transform(melspec)
        return melspec, path

class MultiResolutionDataset(Dataset):
    """Pytorch dataset object for loading processed (i.e. in melspec format) nsynth dataset or subset"""

    def __init__(self, select, path, resolution=256):
        self.audiolist = get_selection(select, path)
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        self.specgrams_helper = spec.SpecgramsHelper(audio_length=64000,
                                                     spec_shape=resolution,
                                                     window_length=resolution * 2,
                                                     sample_rate=16000,
                                                     mel_downscale=1,
                                                     ifreq=True,
                                                     )

    def __len__(self):
        return len(self.audiolist)

    def __getitem__(self, idx):
        path = self.audiolist[idx]
        sample = pd.read_feather(path)
        audio = sample['audio'][0]
        melspec = self.specgrams_helper.wave_to_normalized_melspecgram(audio).numpy()
        #melspec = np.transpose(melspec, (2, 0, 1))
        melspec = self.transform(melspec)
        return melspec


class MultiLabelResolutionDataset(Dataset):
    def __init__(self, select, path, resolution=8):
        self.audiolist = get_selection(select, path)
        self.attributes = [i for i in range(self.__len__())]
        c = list(zip(self.audiolist, self.attributes))
        random.shuffle(c)
        self.audiolist2, self.att2 = zip(*c)
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.RandomHorizontalFlip(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        self.specgrams_helper = spec.SpecgramsHelper(audio_length=64000,
                                                     spec_shape=resolution,
                                                     window_length=resolution * 2,
                                                     sample_rate=16000,
                                                     mel_downscale=1,
                                                     ifreq=True,
                                                     )

    def __len__(self):
        return len(self.audiolist)

    def __getitem__(self, index):
        path = self.audiolist[index]
        sample = pd.read_feather(path)
        audio = sample['audio'][0]
        melspec = self.specgrams_helper.wave_to_normalized_melspecgram(audio).numpy()
        melspec = self.transform(melspec)
        label_org = self.attributes[index]
        label_trg = self.att2[index]
        return melspec, label_org, label_trg


class MultiLabelAllDataset(Dataset):
    def __init__(self, select, path, resolution=256):

        self.audiolist = get_selection(select, path)
        self.slist = [os.path.basename(file).split('_')[0] for file in self.audiolist]
        self.plist = [file.split('_')[2].split('-')[0] for file in self.audiolist]
        self.attributes = [i for i in range(self.__len__())]
        c = list(zip(self.audiolist, self.slist, self.plist, self.attributes))
        random.shuffle(c)
        self.audiolist2, self.slist2, self.plist2, self.att2 = zip(*c)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        self.specgrams_helper = spec.SpecgramsHelper(audio_length=64000,
                                                     spec_shape=resolution,
                                                     window_length=resolution * 2,
                                                     sample_rate=16000,
                                                     mel_downscale=1,
                                                     ifreq=True,
                                                     )

    def __len__(self):
        return len(self.slist)

    def __getitem__(self, index):
        path = self.audiolist[index]
        sample = pd.read_feather(path)
        audio = sample['audio'][0]
        melspec = self.specgrams_helper.wave_to_normalized_melspecgram(audio).numpy()
        melspec = np.transpose(melspec, (2, 0, 1))
        melspec = self.transform(melspec)
        sty = self.slist[index]
        sty.requires_grad = False
        pix = self.plist[index]
        pix.requires_grad = False

        label_org = self.attributes[index]
        label_trg = self.att2[index]
        return melspec, sty, pix, label_org, label_trg