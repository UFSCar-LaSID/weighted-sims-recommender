from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import implicit
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
import pickle 
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
    
    def get_user_index(self, user_id):
        return self.le_users.transform(user_id)
    
    def get_item_index(self, item_id):
        return self.le_items.transform(item_id)
    
    def get_user_id(self, user_index):
        return self.le_users.inverse_transform(user_index)
    
    def get_item_id(self, item_index):
        return self.le_items.inverse_transform(item_index)


class ImplicitRecommender(object):

    def _save_embeddings(self):
        item_embeddings = self.model.item_factors if kw.TRAIN_MODE == 'cpu' else self.model.item_factors.to_numpy()
        user_embeddings = self.model.user_factors if kw.TRAIN_MODE == 'cpu' else self.model.item_factors.to_numpy()
        np.save(os.path.join(self.embeddings_filepath, 'items.npy'), item_embeddings)
        np.save(os.path.join(self.embeddings_filepath, 'users.npy'), user_embeddings)
        pickle.dump(self.sparse_repr, open(os.path.join(self.embeddings_filepath, 'sparse_repr.pkl'), 'wb'))


    def fit(self, df_train):
        self.sparse_repr = SparseRepr(df_train)
        self.train_user_items_matrix = self.sparse_repr.get_user_items_matrix(df_train)
        self.model.fit(self.train_user_items_matrix, show_progress=False)
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
            np.vstack([
                np.repeat(userid, kw.TOP_N), 
                self.sparse_repr.get_item_id(np.ravel(recommended_items))]  # Para que os itens sejam id's e não indexes
            ).T,
            columns=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]
        )
        return recommendations
    

class ALS(ImplicitRecommender):
    def __init__(self, embeddings_filepath, factors=64, regularization=0.01, iterations=15):
        self.embeddings_filepath = embeddings_filepath        
        ALSModel = implicit.cpu.als.AlternatingLeastSquares if kw.TRAIN_MODE == 'cpu' else implicit.gpu.als.AlternatingLeastSquares
        self.model = ALSModel(factors=factors, regularization=regularization, iterations=iterations, random_state=kw.RANDOM_STATE)


class BPR(ImplicitRecommender):
    def __init__(self, embeddings_filepath, factors=100, learning_rate=0.01, regularization=0.01, iterations=100):
        self.embeddings_filepath = embeddings_filepath
        BPRModel = implicit.cpu.bpr.BayesianPersonalizedRanking if kw.TRAIN_MODE == 'cpu' else implicit.gpu.bpr.BayesianPersonalizedRanking        
        self.model = BPRModel(factors=factors, learning_rate=learning_rate, regularization=regularization, iterations=iterations, random_state=kw.RANDOM_STATE)



class GensimRecommender(object):

    def __init__(self, embeddings_filepath, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=1e-3):
        self.embeddings_filepath = embeddings_filepath
        self.hyperparams = {
            'size': embedding_dim,
            'epochs': n_epochs,
            'negative': negative_sampling,
            'ns_exponent': negative_exponent,
            'sample': (0 if subsampling_p is None else subsampling_p)
        }

    def _save_embeddings(self):
        item_embeddings = self.model.wv.vectors[np.argsort(np.fromiter(self.model.wv.index2word, dtype=np.int32, count=len(self.model.wv.index2word)))]
        np.save(os.path.join(self.embeddings_filepath, 'items.npy'), item_embeddings)
        if self.has_user_embeddings:
            user_embeddings = self.model.docvecs.vectors_docs        
            np.save(os.path.join(self.embeddings_filepath, 'users.npy'), user_embeddings)
        pickle.dump(self.sparse_repr, open(os.path.join(self.embeddings_filepath, 'sparse_repr.pkl'), 'wb'))

    def _generate_interactions_file(self):
        self.interactions_file = 'interactions.temp'
        with open(self.interactions_file, 'w') as f:
            for user in range(self.train_user_items_matrix.shape[0]):
                f.write(' '.join(self.train_user_items_matrix[user].nonzero()[1].astype(str)) + '\n')
    
    def _remove_interactions_file(self):
        os.remove(self.interactions_file)

    def fit(self, df_train):
        self.sparse_repr = SparseRepr(df_train)
        self.train_user_items_matrix = self.sparse_repr.get_user_items_matrix(df_train)
        self._generate_interactions_file()
        self.model = self.GensimModel(
            **self.hyperparams,
            **self.model_params,
            corpus_file=self.interactions_file,
            window=self.train_user_items_matrix.sum(axis=1).max()*10000,
            min_count=1,
            workers=cpu_count(),
            hs=0,
            max_vocab_size=None,
            trim_rule=None,
            seed=2023
        )
        self._save_embeddings()
        self._remove_interactions_file() 


class Item2Vec(GensimRecommender):
    def __init__(self, embeddings_filepath, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=0.001):
        super().__init__(embeddings_filepath, embedding_dim, n_epochs, negative_sampling, negative_exponent, subsampling_p)
        self.has_user_embeddings = False
        self.GensimModel = Word2Vec
        self.model_params = {
            'sg': 1,
            'max_final_vocab': None,
            'sorted_vocab': 0,
            'compute_loss': False
        }


class User2Vec(GensimRecommender):
    def __init__(self, embeddings_filepath, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=0.001):
        super().__init__(embeddings_filepath, embedding_dim, n_epochs, negative_sampling, negative_exponent, subsampling_p)
        self.has_user_embeddings = True
        self.GensimModel = Doc2Vec
        self.model_params = {
            'dm': 1,
            'dm_mean': 1,
            'dm_concat': 0            
        }


class Interact2Vec(object):
    def __init__(self, embeddings_filepath, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=0.001):
        self.embeddings_filepath = embeddings_filepath
        self.subsampling_p = subsampling_p

    def _subsample_items(self, df):
        if self.subsampling_p is not None:
            freq = df.groupby(kw.COLUMN_ITEM_ID).size()
            n_interactions = len(df)
            z = freq / n_interactions
            keep_prob = (np.sqrt(z/self.subsampling_p) + 1) * (self.subsampling_p/z)
            keep_prob = keep_prob.reindex(df[rr.COLUMN_ITEM_ID])
            keep_prob.index = df.index
            discarded_interactions = keep_prob < self.rng.rand(n_interactions)
            return df[~discarded_interactions].copy()
        return df.copy()
    
    def sparse_sampling(df, sparse_repr):
        n_interactions = len(df)
        sm = sparse_repr.get_user_items_matrix(df)    
        z = sm.sum(axis=0).A1 / n_interactions    
        keep_prob = (np.sqrt(z/SUBSAMPLING_P) + 1) * (SUBSAMPLING_P/z)    
        mask = sm.multiply(keep_prob).data < np.random.rand(n_interactions)
        sm.data[mask] = 0    
        sm.eliminate_zeros()
        return sm


    def fit(self, df_train):
        self.sparse_repr = SparseRepr(df_train)
        sampled_df = self._subsample_items(df_train)