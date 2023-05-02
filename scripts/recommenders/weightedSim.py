import scripts as kw
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


class WeightedSim(object):
    def __init__(self, embeddings_filepath, k=64, user_item_weights=None, similarity_metric='cosine', **model_params):
        self.k = k
        self.user_item_weights = user_item_weights
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_filepath, kw.FILE_SPARSE_REPR), 'rb'))
        self.item_embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), 'rb'))
        self.user_embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_USERS_EMBEDDINGS), 'rb'))
        self.similarity_metric = similarity_metric
    
    def fit(self, df):
        # Similaridade entre itens
        n_items = self.item_embeddings.shape[0]
        items_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * n_items))
        self.item_item_sim = pd.DataFrame()
        for i in range(0, n_items, items_per_batch):
            batch_items = self.sparse_repr.get_item_id(np.arange(i, min(i+items_per_batch, n_items)))
            if self.similarity_metric == 'cosine':
                batch_sims = cosine_similarity(self.item_embeddings[i:i+items_per_batch], self.item_embeddings)
            elif self.similarity_metric == 'dot':
                batch_sims = np.dot(self.item_embeddings[i:i+items_per_batch], self.item_embeddings.T)
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            self.item_item_sim = pd.concat([
                self.item_item_sim,    
                pd.DataFrame(
                    np.column_stack([
                        np.repeat(batch_items, self.k),
                        self.sparse_repr.get_item_id(np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k].flatten()),
                        np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k].flatten(),
                    ]),
                    columns=[kw.COLUMN_ITEM_ID, 'neighbor', 'sim']            
                )
            ])
        self.df_train = df.copy()
    
    def recommend(self, df):
        target_users = sorted(df[kw.COLUMN_USER_ID].unique())
        user_item_sim = pd.DataFrame()
        users_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * len(df)))
        for u in range(0, len(target_users), users_per_batch):
            batch_users = target_users[u:u+users_per_batch]
            batch_encoder = LabelEncoder()
            batch_encoder.fit(batch_users)
            batch_users = batch_encoder.inverse_transform(np.arange(len(batch_users)))
            users_idx = self.sparse_repr.get_user_index(batch_users)
            if self.similarity_metric == 'cosine':
                batch_sims = cosine_similarity(self.user_embeddings[users_idx], self.item_embeddings)
            elif self.similarity_metric == 'dot':
                batch_sims = np.dot(self.user_embeddings[users_idx], self.item_embeddings.T)            
            known_interactions = self.df_train[self.df_train[kw.COLUMN_USER_ID].isin(batch_users)]
            batch_sims[
                batch_encoder.transform(known_interactions[kw.COLUMN_USER_ID]), 
                self.sparse_repr.get_item_index(known_interactions[kw.COLUMN_ITEM_ID].values)
            ] = -np.inf
            user_item_sim = pd.concat([
                user_item_sim,    
                pd.DataFrame(
                    np.column_stack([
                        np.repeat(batch_users, self.k),
                        self.sparse_repr.get_item_id(np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k].flatten()),
                        np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k].flatten(),
                    ]),
                    columns=[kw.COLUMN_USER_ID, 'neighbor', 'sim']
                )
            ])
        user_item_sim = user_item_sim.set_index([kw.COLUMN_USER_ID, 'neighbor'])['sim']
        item_based_neighborhood = pd.merge(self.df_train[self.df_train[kw.COLUMN_USER_ID].isin(target_users)], self.item_item_sim, on=kw.COLUMN_ITEM_ID, how='inner')        
        item_based_neighborhood_qt = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor']).size()
        if self.user_item_weights is None:
            item_based_neighborhood_sim = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor'])['sim'].sum()
            final_sim = item_based_neighborhood_sim.add(user_item_sim, fill_value=0).divide(item_based_neighborhood_qt.add(pd.Series(1, index=user_item_sim.index), fill_value=0)).to_frame('sim').reset_index()
        else:
            user_weight, item_weight = self.user_item_weights
            item_based_neighborhood_sim = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor'])['sim'].mean().multiply(item_weight)
            final_sim = item_based_neighborhood_sim.add(user_item_sim.multiply(user_weight), fill_value=0).divide(item_weight+user_weight).to_frame('sim').reset_index()        
        del item_based_neighborhood
        del item_based_neighborhood_sim
        del item_based_neighborhood_qt
        final_sim = final_sim.merge(
            self.df_train, 
            how='left', 
            left_on=[kw.COLUMN_USER_ID, 'neighbor'], 
            right_on=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]
        )
        final_sim = final_sim[final_sim[kw.COLUMN_ITEM_ID].isna()].drop(columns=[kw.COLUMN_ITEM_ID])
        recommendations = final_sim.sort_values('sim', ascending=False).groupby(kw.COLUMN_USER_ID).head(kw.TOP_N).sort_values(['id_user', 'sim'], ascending=[True, False])
        del final_sim
        
        recommendations[kw.COLUMN_RANK] = np.concatenate(recommendations.groupby(kw.COLUMN_USER_ID).size().sort_index(ascending=True).apply(lambda x:np.arange(1, x+1)).values)
        recommendations = recommendations.rename(columns={'neighbor': kw.COLUMN_ITEM_ID})[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RANK]].reset_index(drop=True)
        return recommendations