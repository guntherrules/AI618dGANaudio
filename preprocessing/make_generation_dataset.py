from sklearn.model_selection import train_test_split
import glob
import os
import pandas as pd
import shutil

files = glob.glob('../data/all_samples_512/train_512/'+'*npy')
instrument_families = [os.path.basename(file).split('_')[0] for file in files]
data = pd.DataFrame({'file': files, 'instrument_family': instrument_families})
train, test = train_test_split(data, shuffle=True, stratify=data['instrument_family'])
print(train['instrument_family'].value_counts())
print(train[train['instrument_family'] == 'string'])
for instrument in set(train['instrument_family'].tolist()):
    path = '/generator_512/' + instrument
    subset = train[train['instrument_family'] == instrument]
    for file in subset['file']:
        save_path = file.replace('/all_samples_512/train_512', path)
        shutil.copy(file, save_path)


