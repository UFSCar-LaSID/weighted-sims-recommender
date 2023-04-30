import pandas as pd

import scripts as kw
from scripts.recommenders.mf import ALS, BPR
from scripts.recommenders.hyperparameters import ALS_HYPERPARAMETERS, BPR_HYPERPARAMETERS, ALS_ITEM_SIM_HYPERPARAMETERS, BPR_ITEM_SIM_HYPERPARAMETERS
from scripts.recommenders.itemSim import ItemSim


RECOMMENDERS_TABLE = pd.DataFrame(
    [[1,  'ALS',          'ALS',      ALS,      ALS_HYPERPARAMETERS],
     [2,  'BPR',          'BPR',      BPR,      BPR_HYPERPARAMETERS],
     [3,  'ALS_itemSim',  'ALS',      ItemSim,  ALS_ITEM_SIM_HYPERPARAMETERS],
     [4,  'BPR_itemSim',  'BPR',      ItemSim,  BPR_ITEM_SIM_HYPERPARAMETERS]], 
    columns=[kw.RECOMMENDER_ID, kw.RECOMMENDER_NAME, kw.RECOMMENDER_EMBEDDINGS, kw.RECOMMENDER_CLASS, kw.RECOMMENDER_HYPERPARAMETERS]
).set_index(kw.RECOMMENDER_ID)

class Recommender(object):
    def __init__(self, recommender_id):
        self.name = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_NAME]
        self.embeddings_name = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_EMBEDDINGS]
        self.model = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_CLASS]
        self.hyperparameters = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_HYPERPARAMETERS]

    def get_name(self):
        return self.name
    
    def get_embeddings_name(self):
        return self.embeddings_name
    
    def get_model(self):
        return self.model
    
    def get_hyperparameters(self):
        return self.hyperparameters
    


def get_recommenders(recommenders=None):
    for recommender_id, recommender_data in RECOMMENDERS_TABLE.iterrows():
        if recommenders is None or recommender_data[kw.RECOMMENDER_NAME] in recommenders:
            yield Recommender(recommender_id)