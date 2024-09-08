import src as kw
import pickle
import os

import numpy as np
import turicreate as tc
from sklearn.metrics.pairwise import cosine_similarity


class ItemSim(object):
    def __init__(self, embeddings_filepath, k=kw.K, **model_params):        
        self.k = k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_filepath, kw.FILE_SPARSE_REPR), 'rb'))        
        self.embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), 'rb'))
    
    def get_items_sims(self):
        df = self.sims.to_dataframe()
        df.columns = [kw.COLUMN_ITEM_ID, 'similar_id_item', 'similarity']
        return df


    def fit(self, df_train):
        n_items = self.embeddings.shape[0]
        items_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * n_items))
        nearest_neighbors = np.empty((n_items, self.k))
        nearest_sims = np.empty((n_items, self.k))
        embeddings_norm = self.embeddings / np.sqrt(np.sum(self.embeddings**2, axis=1)).reshape(-1,1) # Normaliza embeddings
        for i in range(0, n_items, items_per_batch):
            batch_sims = np.dot(embeddings_norm[i:i+items_per_batch], embeddings_norm.T) # Calcula distancia
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            nearest_neighbors[i:i+items_per_batch] = np.argpartition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k] # captura k mais similares
            nearest_sims[i:i+items_per_batch] = -np.partition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k] # captura similaridades dos k vizinhos
        sim_table = tc.SFrame({
            'id_item': self.sparse_repr.get_item_id(np.repeat(np.arange(n_items), self.k).astype(int)),
            'similar': self.sparse_repr.get_item_id(nearest_neighbors.flatten().astype(int)),
            'score': nearest_sims.flatten()
        })
        self.sims = sim_table
        sframe = tc.SFrame(df_train[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]])
        self.model = tc.recommender.item_similarity_recommender.create(
            observation_data=sframe,
            user_id=kw.COLUMN_USER_ID,
            item_id=kw.COLUMN_ITEM_ID,
            similarity_type='cosine',
            only_top_k=self.k,
            nearest_items=sim_table,
            target_memory_usage=kw.MEM_SIZE_LIMIT,
            verbose=False
        )
    
    def recommend(self, df_test):
        recommendations = self.model.recommend(
            users=df_test[kw.COLUMN_USER_ID].unique(),
            k=kw.TOP_N,
            exclude_known=True
        ).to_dataframe().drop(columns=['score'])
        return recommendations

