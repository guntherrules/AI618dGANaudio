from sklearn.model_selection import train_test_split
import glob
import os
import pandas as pd
import shutil

def move_files(files, dest_paths):
    for file, dest in zip(files,dest_paths):
        shutil.move(file,dest)

files = glob.glob('../data/all_samples_512/'+'*npy')
instrument_families = [os.path.basename(file).split('_')[0] for file in files]
data = pd.DataFrame({'file': files, 'instrument_family': instrument_families})
train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['instrument_family'])
train_dest = [path.replace('all_samples_512','all_samples_512/train') for path in train['file']]
test_dest = [path.replace('all_samples_512','all_samples_512/test') for path in test['file']]
move_files(train['file'],train_dest)
move_files(test['file'],test_dest)
