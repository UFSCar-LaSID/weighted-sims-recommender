import tqdm
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

from scripts.dataset import get_datasets
import scripts as kw

DATASETS = ['RetailRocket-Transactions']

SUBSAMPLING_P = 0.0001

class SparseRepr(object):
    def __init__(self, df):
        self.le_users = LabelEncoder()
        self.le_users.fit(df[kw.COLUMN_USER_ID])
        self.le_items = LabelEncoder()
        self.le_items.fit(df[kw.COLUMN_ITEM_ID])

    def get_user_items_matrix(self, df):
        data = np.ones(len(df))
        user_ind = self.le_users.transform(df[kw.COLUMN_USER_ID])
        item_ind = self.le_items.transform(df[kw.COLUMN_ITEM_ID])
        n_users = len(self.le_users.classes_)
        n_items = len(self.le_items.classes_)
        return csr_matrix((data, (user_ind, item_ind)), shape=(n_users, n_items))
    
    def get_user_index(self, user_id):
        return self.le_users.transform(user_id)
    
    def get_item_index(self, item_id):
        return self.le_items.transform(item_id)
    
    def get_user_id(self, user_index):
        return self.le_users.inverse_transform(user_index)
    
    def get_item_id(self, item_index):
        return self.le_items.inverse_transform(item_index)
    

def pandas_sampling(df, sparse_repr):
    freq = df.groupby(kw.COLUMN_ITEM_ID).size()
    n_interactions = len(df)
    z = freq / n_interactions
    keep_prob = (np.sqrt(z/SUBSAMPLING_P) + 1) * (SUBSAMPLING_P/z)
    keep_prob = keep_prob.reindex(df[kw.COLUMN_ITEM_ID])
    keep_prob.index = df.index
    discarded_interactions = keep_prob < np.random.rand(n_interactions)
    sm = sparse_repr.get_user_items_matrix(df[~discarded_interactions].copy())
    return sm

def sparse_sampling(df, sparse_repr):
    n_interactions = len(df)
    sm = sparse_repr.get_user_items_matrix(df)    
    z = sm.sum(axis=0).A1 / n_interactions    
    keep_prob = (np.sqrt(z/SUBSAMPLING_P) + 1) * (SUBSAMPLING_P/z)    
    mask = sm.multiply(keep_prob).data < np.random.rand(n_interactions)
    sm.data[mask] = 0    
    sm.eliminate_zeros()
    return sm


for dataset in get_datasets(datasets=DATASETS):
    df = dataset.get_dataframe()    
    sparse_repr = SparseRepr(df)

    pandas_time = list()
    sparse_time = list()

    for _ in tqdm.tqdm(range(500)):
        start = time.time()
        pandas_sampling(df, sparse_repr)
        end = time.time()
        pandas_time.append(end-start)

        start = time.time()
        sparse_sampling(df, sparse_repr)
        end = time.time()
        sparse_time.append(end-start)

    print('Pandas: ', np.mean(pandas_time))
    print('Sparse: ', np.mean(sparse_time))
