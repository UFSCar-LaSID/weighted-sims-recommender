from implicit.cpu.als import AlternatingLeastSquares
from implicit.gpu.bpr import BayesianPersonalizedRanking
import numpy as np
import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

import scripts as kw

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
    
    def get_user_index(self, userid):
        return self.le_users.transform(userid)


class ImplicitRecommender(object):

    def _save_embeddings(self):
        item_embeddings = self.model.item_factors
        user_embeddings = self.model.user_factors
        np.save(os.path.join(self.embeddings_filepath, 'items.npy'), item_embeddings)
        np.save(os.path.join(self.embeddings_filepath, 'users.npy'), user_embeddings)


    def fit(self, df_train):
        self.sparse_repr = SparseRepr(df_train)
        self.train_user_items_matrix = self.sparse_repr.get_user_items_matrix(df_train)
        self.model.fit(self.train_user_items_matrix, show_progress=True)
        self._save_embeddings()


    def recommend(self, df_test):
        userid = df_test[kw.COLUMN_USER_ID].unique()
        users_sparse_index = self.sparse_repr.get_user_index(userid)        
        recommended_items, _ = self.model.recommend(
            userid=users_sparse_index, 
            user_items=self.train_user_items_matrix[users_sparse_index, :], 
            N=kw.TOP_N,
            filter_already_liked_items=True
        )
        recommendations = pd.DataFrame(
            np.vstack([np.repeat(userid, kw.TOP_N), np.ravel(recommended_items)]).T,
            columns=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]
        )
        return recommendations
    

class ALS(ImplicitRecommender):
    def __init__(self, embeddings_filepath, random_state, factors=64, regularization=0.01, iterations=15):
        self.embeddings_filepath = embeddings_filepath
        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations, random_state=random_state)


class BPR(ImplicitRecommender):
    def __init__(self, embeddings_filepath, random_state, factors=100, learning_rate=0.01, regularization=0.01, iterations=100):
        self.embeddings_filepath = embeddings_filepath
        self.model = BayesianPersonalizedRanking(factors=factors, learning_rate=learning_rate, regularization=regularization, iterations=iterations, random_state=random_state)