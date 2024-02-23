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
        if similarity_metric == 'cosine': # verifica se sera similaridade de cosseno
            self.item_embeddings = self.item_embeddings / np.sqrt(np.sum(self.item_embeddings**2, axis=1)).reshape(-1,1) # normaliza embeddings de itens

    def fit(self, df):        
        pass
    
    def recommend(self, df):
        pass