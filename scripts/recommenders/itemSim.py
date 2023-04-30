import scripts as kw
import pickle
import os
import numpy as np
import turicreate as tc
from sklearn.metrics.pairwise import cosine_similarity


class ItemSim(object):
    def __init__(self, embeddings_filepath, k=64, **model_params):        
        self.k = k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_filepath, 'sparse_repr.pkl'), 'rb'))        
        self.embeddings = np.load(open(os.path.join(embeddings_filepath, 'items.npy'), 'rb'))
    
    def get_items_sims(self):
        df = self.sims.to_dataframe()
        df.columns = [kw.COLUMN_ITEM_ID, 'similar_id_item', 'similarity']
        return df


    def fit(self, df_train):
        n_items = self.embeddings.shape[0]
        items_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * n_items))
        nearest_neighbors = np.empty((n_items, self.k))
        nearest_sims = np.empty((n_items, self.k))
        for i in range(0, n_items, items_per_batch):
            batch_sims = cosine_similarity(self.embeddings[i:i+items_per_batch], self.embeddings)
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            nearest_neighbors[i:i+items_per_batch] = np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k]
            nearest_sims[i:i+items_per_batch] = np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k]
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

