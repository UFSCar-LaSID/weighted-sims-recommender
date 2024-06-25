import scripts as kw
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


class MeanSim(object):
    def __init__(self, embeddings_filepath, k=kw.K, similarity_metric='cosine', **model_params):
        self.k = k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_filepath, kw.FILE_SPARSE_REPR), 'rb'))
        self.item_embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), 'rb'))

    def fit(self, df):
        
        self.df_train = df
        mean_item_sim = pd.DataFrame()
        
        user_item_interaction = self.sparse_repr.get_user_items_matrix(df)

        usuario_ids, item_ids = np.nonzero(user_item_interaction)
        user_avg_embeddings = np.zeros((user_item_interaction.shape[0], self.item_embeddings.shape[1]))
        np.add.at(user_avg_embeddings, usuario_ids, self.item_embeddings[item_ids])
        user_avg_embeddings /= np.bincount(usuario_ids).reshape(-1, 1)

        user_avg_embeddings_norm = user_avg_embeddings / np.sqrt(np.sum(user_avg_embeddings**2, axis=1)).reshape(-1, 1)
        self.item_embeddings_norm = self.item_embeddings / np.sqrt(np.sum(self.item_embeddings**2, axis=1)).reshape(-1, 1)

        target_users = df[kw.COLUMN_USER_ID].unique() 
        n_users = len(user_avg_embeddings)

        self.recommendations = pd.DataFrame()
        users_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * n_users))       
        for u in range(0, n_users, users_per_batch):

            batch_users = target_users[u:u+users_per_batch]
            batch_encoder = LabelEncoder()
            batch_encoder.fit(batch_users)
            users_idx = self.sparse_repr.get_user_index(batch_users)

            batch_sims = np.dot(user_avg_embeddings_norm[users_idx], self.item_embeddings_norm.T)

            known_interactions = self.df_train[self.df_train[kw.COLUMN_USER_ID].isin(batch_users)]
            batch_sims[
                batch_encoder.transform(known_interactions[kw.COLUMN_USER_ID]), 
                self.sparse_repr.get_item_index(known_interactions[kw.COLUMN_ITEM_ID].values)
            ] = -np.inf

            self.recommendations = pd.concat([
                self.recommendations,
                pd.DataFrame(
                    np.column_stack([
                        np.repeat(batch_users, self.k),
                        self.sparse_repr.get_item_id(np.argsort(-batch_sims, axis=1)[:, :self.k].flatten()), 
                        -np.sort(-batch_sims, axis=1)[:, :self.k].flatten()                         
                    ]),
                    columns=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, 'sim']
                )
            ])

    def recommend(self, df):

        mask = self.recommendations[kw.COLUMN_USER_ID].isin(df[kw.COLUMN_USER_ID].unique())
        self.recommendations = self.recommendations.loc[mask]

        self.recommendations.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.recommendations.dropna(inplace=True)

        self.recommendations[kw.COLUMN_RANK] = np.concatenate(self.recommendations.groupby(kw.COLUMN_USER_ID).size().sort_index(ascending=True).apply(lambda x:np.arange(1, x+1)).values)
        self.recommendations = self.recommendations.astype({
            kw.COLUMN_ITEM_ID: 'int64',
            kw.COLUMN_USER_ID: 'int64'
        })

        return self.recommendations