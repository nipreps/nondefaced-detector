import os, sys
sys.path.append('..')
from defacing.helpers.utils import load_vol

import numpy as np
import pandas as pd
from glob import glob

orig_data_face = '/work/01329/poldrack/data/mriqc-net/face/T1w'
orig_data_deface = '/work/01329/poldrack/data/mriqc-net/defaced'

regis_data_face = '/work/06850/sbansal6/maverick2/mriqc-shared/conformed/face'#'/work/06595/kavinash/mriqc-shared-directory/Registered/face'
regis_data_deface ='/work/06850/sbansal6/maverick2/mriqc-shared/conformed/deface'# '/work/06595/kavinash/mriqc-shared-directory/Registered/defaced'


faced_path = []
for pth in glob(regis_data_face + '/*/*.nii.gz'):
    if not pth.__contains__('edge'):
        try:
            load_vol(pth)
            faced_path.append(pth)
        except: pass

defaced_path = []
for pth in glob(regis_data_deface + '/*/*.nii.gz'):
    if not pth.__contains__('edge'):
        try:
            load_vol(pth)
            defaced_path.append(pth)
        except: pass


save_path = '/work/06850/sbansal6/maverick2/mriqc-shared/csv/'

os.makedirs(save_path, exist_ok=True)
path = []
label = []

path.extend(faced_path)
label.extend([1]*len(faced_path))

path.extend(defaced_path)
label.extend([0]*len(defaced_path))

df = pd.DataFrame()
df['X'] = path
df['Y'] = label
df.to_csv(os.path.join(save_path, 'all.csv'))

from operator import itemgetter 
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


SPLITS = 10
skf = StratifiedKFold(n_splits=SPLITS)
fold_no = 1

for train_index, test_index in skf.split(path, label):
    out_path = save_path + '/train_test_fold_{}/csv/'.format(fold_no)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    image_train, image_test = itemgetter(*train_index)(path), itemgetter(*test_index)(path) 
    label_train, label_test = itemgetter(*train_index)(label), itemgetter(*test_index)(label) 
    
    # image_train = [os.path.join(data_path, 'sub-' + str(pth) + '_T1w.nii.gz') for pth in image_train]
    train_data = {'X': image_train, 'Y': label_train}
    df_train = pd.DataFrame(train_data)
    df_train.to_csv(os.path.join(out_path, 'training.csv'), index = False)

    # image_test = [os.path.join(data_path, 'sub-' + str(pth) + '_T1w.nii.gz') for pth in image_test]
    validation_data = {'X': image_test, 'Y': label_test}
    df_validation = pd.DataFrame(validation_data)
    df_validation.to_csv(os.path.join(out_path, 'validation.csv'), index = False)
 
    fold_no += 1   