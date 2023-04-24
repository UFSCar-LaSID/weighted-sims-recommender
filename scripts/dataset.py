import csv
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import os
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

import scripts as kw

DATASETS_TABLE = pd.DataFrame(
    [[1,  'Anime Recommendations',     'E', '1GSz8EKsA3JlKfI-4qET0nUtmoyBGWNJl'],
     [2,  'BestBuy',                   'I', '1WZY5i6rRTBH4g8M5Qd0oSBVcWbis14Zq'],
     [3,  'Book-Crossing',             'E', '1mFC20Rauj-PRhYNm_jzzDGmKafobWdrq'],
     [4,  'CiaoDVD',                   'E', '1a_9fVVrelz-8XYs3tHZM8rnLtZRe6x8H'],
     [5,  'DeliciousBookmarks',        'I', '14geC9mUx1--xHkAUPtYLrMfZk4jc4ITW'],
     [6,  'Filmtrust',                 'E', '1V9Hd0DLhZzmA6c2bprmUlW6LWjfXC10p'],
     [7,  'Jester',                    'E', '1Yw28A-2l5Z-xB48puSWw4C_oP_DypNry'],
     [8,  'Last.FM - Listened',        'I', '1g3j9UP2a0gvB0fYJ9OzPAW1k1g59JobH'],
     [9,  'Last.FM - Tagged',          'I', '1bDHeh7L2TbBC_hJJCrb2TSohhK2Hl6ah'],
     [10, 'LibimSeTi',                 'E', '1AtmEnX415YAlUPmqjOK5pXGoJBSO3Jt3'],
     [11, 'MovieLens',                 'E', '1Tbi5EVs7BBZmnuKaFHDZelFgDuz-9YEP'],
     [12, 'NetflixPrize',              'E', '1gpoUoSFQTTAIUtdYCVLRuv0SI6vAZifr'],
     [13, 'RetailRocket-All',          'I', '12oHsCzjrlNbRe_pvTOCxkWVFlfO9WjhH'],
     [14, 'RetailRocket-Transactions', 'I', '12EwJisOE-6-xvYXe-YN_qX1FBfFWEiZv']], 
    columns=[kw.DATASET_ID, kw.DATASET_NAME, kw.DATASET_TYPE, kw.DATASET_GDRIVE]
)

# class SparseRepr(object):
#     def __init__(self, df=None, users=None, items=None):
#         if df is not None and users is None or items is None:
#             users = df[kw.COLUMN_USER_ID].unique()
#             items = df[kw.COLUMN_ITEM_ID].unique()
#         if users is not None and items is not None:
#             self._create_encoders(users, items)
#         else:
#             raise Exception('Error: wrong parameters for class SparseRepr.')

#     def _create_encoders(self, users, items):
#         self._user_encoder = LabelEncoder()
#         self._item_encoder = LabelEncoder()
#         self._user_encoder.fit(users)
#         self._item_encoder.fit(items)

#     def get_matrix(self, users, items, interactions=None):
#         # Captura os indices de linha e coluna
#         users_coo, items_coo = self._user_encoder.transform(users), self._item_encoder.transform(items)
#         # Constroi o vetor de conteudo da matrix
#         data = interactions if interactions is not None else np.ones(len(users_coo))
#         # Constroi a matriz
#         sparse_matrix = sparse.coo_matrix((data, (users_coo, items_coo)), shape=(len(self._user_encoder.classes_), len(self._item_encoder.classes_))).tocsr()
#         return sparse_matrix

#     def get_n_users(self):
#         return len(self._user_encoder.classes_)

#     def get_n_items(self):
#         return len(self._item_encoder.classes_)

#     def get_idx_of_user(self, user):
#         return self._user_encoder.transform(user) if type(user) in [list, np.ndarray] else self._user_encoder.transform([user])[0]

#     def get_idx_of_item(self, item):
#         return self._item_encoder.transform(item) if type(item)in [list, np.ndarray] else self._item_encoder.transform([item])[0]

#     def get_user_of_idx(self, idx):
#         return self._user_encoder.inverse_transform(idx) if type(idx) in [list, np.ndarray, pd.Series] else self._user_encoder.inverse_transform([idx])[0]

#     def get_item_of_idx(self, idx):
#         return self._item_encoder.inverse_transform(idx) if type(idx) in [list, np.ndarray, pd.Series] else self._item_encoder.inverse_transform([idx])[0]


class Dataset(object):

    def __init__(self, name, path):
        self.name = name
        self.df = pd.read_csv(path, delimiter=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=0)
        self.df = self.df.dropna().drop_duplicates(subset=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID], keep='last')
        if kw.COLUMN_RATING in self.df.columns:
            explicit_ratings = self.df[kw.COLUMN_RATING]!=-1
            min_max = self.df[explicit_ratings][kw.COLUMN_RATING].apply(['min', 'max'])
            mean_rating = min_max.loc['min'] + (min_max.loc['max']-min_max.loc['min'])/2            
            self.df = self.df[(self.df[kw.COLUMN_RATING]>=mean_rating)|(self.df[kw.COLUMN_RATING]==-1)]        
   
        # if self.name == 'Anime Recommendations':
        #     df = df[df[rr.COLUMN_INTERACTION]!=-1] if ds_type == 'E' else df[df[rr.COLUMN_INTERACTION]==-1]
        # elif self.name == 'Book-Crossing':
        #     df = df[df[rr.COLUMN_INTERACTION]!=0] if ds_type == 'E' else df[df[rr.COLUMN_INTERACTION]==0]           
    
    def get_name(self):
        return self.name

    def get_dataframe(self):
        return self.df

    def get_n_users(self):
        return self.df[kw.COLUMN_USER_ID].nunique()

    def get_n_items(self):
        return self.df[kw.COLUMN_ITEM_ID].nunique()

    def get_n_interactions(self):
        return len(self.df)


# # Faz o download dos datasets
# def download_datasets(ds_dir='datasets', datasets=None, verbose=True):
#     downloaded_datasets = list()
#     d = 0
#     for _, row in DATASETS_TABLE.iterrows():
#         cur_dataset = row[kw.DATASET_NAME]
#         if datasets is None or cur_dataset in datasets:
#             d += 1
#             if verbose:
#                 print('Downloading {}... ({}/{})'.format(cur_dataset, d, len(datasets) if datasets is not None else len(DATASETS_TABLE)))
#             gdd.download_file_from_google_drive(file_id=row[kw.DATASET_GDRIVE], dest_path='./{}/{}.zip'.format(ds_dir, cur_dataset), unzip=False)
#             # os.remove('./{}/{}.zip'.format(ds_dir, cur_dataset))
#             downloaded_datasets.append(cur_dataset)
#     if verbose:
#         print('Datasets downloaded!')
#     return downloaded_datasets


# Recupera um conjunto de datasets, retornando um de cada vez
def get_datasets(dataset_folder='datasets', datasets=None):
    for dataset_name in DATASETS_TABLE[kw.DATASET_NAME]:
        if datasets is None or dataset_name in datasets:
            dataset_filepath = os.path.join(dataset_folder, dataset_name, kw.FILE_INTERACTIONS)
            yield Dataset(dataset_name, dataset_filepath)