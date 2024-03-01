import scripts as kw
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


class WeightedSim(object):
    def __init__(self, embeddings_filepath, k=kw.K, similarity_weights=(0.5, 0.5), similarity_metric='cosine', **model_params):
        self.k = k
        self.user_weight, self.item_weight = similarity_weights # captura o peso das similaridades user-item e item-item       
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_filepath, kw.FILE_SPARSE_REPR), 'rb'))
        self.item_embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), 'rb'))
        self.user_embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_USERS_EMBEDDINGS), 'rb'))        
        if similarity_metric == 'cosine': # verifica se sera similaridade de cosseno
            self.item_embeddings = self.item_embeddings / np.sqrt(np.sum(self.item_embeddings**2, axis=1)).reshape(-1,1) # normaliza embeddings de itens
            self.user_embeddings / np.sqrt(np.sum(self.user_embeddings**2, axis=1)).reshape(-1,1) # normaliza embeddings de usuarios

    
    def fit(self, df):

        n_items = self.item_embeddings.shape[0]
        items_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * n_items))
        self.item_item_sim = pd.DataFrame()        
        for i in range(0, n_items, items_per_batch):
            batch_items = self.sparse_repr.get_item_id(np.arange(i, min(i+items_per_batch, n_items)))
            batch_sims = np.dot(self.item_embeddings[i:i+items_per_batch], self.item_embeddings.T) # calcula similaridade            
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            self.item_item_sim = pd.concat([
                self.item_item_sim,    
                pd.DataFrame(
                    np.column_stack([
                        np.repeat(batch_items, self.k),
                        self.sparse_repr.get_item_id(np.argpartition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k].flatten()), # captura os itens vizinhos
                        -np.partition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k].flatten() # captura similaridades dos k vizinhos
                    ]),
                    columns=[kw.COLUMN_ITEM_ID, 'neighbor', 'sim']
                )
            ])
        self.df_train = df.copy()

    
    def recommend(self, df):
        target_users = df[kw.COLUMN_USER_ID].unique() # remocao da ordenacao para acelerar o codigo
        n_users = len(target_users) # captura numero de usuarios
        user_item_sim = pd.DataFrame()
        users_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * n_users))        
        for u in range(0, n_users, users_per_batch):
            batch_users = target_users[u:u+users_per_batch]
            batch_encoder = LabelEncoder()
            batch_encoder.fit(batch_users)
            # batch_users = batch_encoder.inverse_transform(np.arange(len(batch_users)))
            users_idx = self.sparse_repr.get_user_index(batch_users)
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
                        self.sparse_repr.get_item_id(np.argpartition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k].flatten()), # captura os itens vizinhos
                        -np.partition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k].flatten() # captura similaridades dos k vizinhos                        
                    ]),
                    columns=[kw.COLUMN_USER_ID, 'neighbor', 'sim']
                )
            ])
        user_item_sim = user_item_sim.set_index([kw.COLUMN_USER_ID, 'neighbor'])['sim']

        item_based_neighborhood = pd.merge(self.df_train[self.df_train[kw.COLUMN_USER_ID].isin(target_users)], self.item_item_sim, on=kw.COLUMN_ITEM_ID, how='inner')        
        #item_based_neighborhood_qt = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor']).size()
        # if self.user_item_weights is None:
        #     item_based_neighborhood_sim = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor'])['sim'].sum()
        #     final_sim = item_based_neighborhood_sim.add(user_item_sim, fill_value=0).divide(item_based_neighborhood_qt.add(pd.Series(1, index=user_item_sim.index), fill_value=0)).to_frame('sim').reset_index()
        # else:        
        item_based_neighborhood_sim = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor'])['sim'].mean().multiply(self.item_weight)
        final_sim = item_based_neighborhood_sim.add(user_item_sim.multiply(self.user_weight), fill_value=0).divide(self.item_weight+self.user_weight).to_frame('sim').reset_index()
        del item_based_neighborhood
        del item_based_neighborhood_sim
        #del item_based_neighborhood_qt
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
        recommendations = recommendations.astype({
            kw.COLUMN_ITEM_ID: 'int64',
            kw.COLUMN_USER_ID: 'int64'
        })

        return recommendations