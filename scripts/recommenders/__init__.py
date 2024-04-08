import pandas as pd

import scripts as kw
from scripts.recommenders.mf import ALS, BPR
from scripts.recommenders.hyperparameters import ALS_HYPERPARAMETERS, BPR_HYPERPARAMETERS, ALS_ITEM_SIM_HYPERPARAMETERS, BPR_ITEM_SIM_HYPERPARAMETERS, ALS_WEIGHTED_SIM_HYPERPARAMETERS, BPR_WEIGHTED_SIM_HYPERPARAMETERS, ALS_MEAN_SIM_HYPERPARAMETERS, BPR_MEAN_SIM_HYPERPARAMETERS
from scripts.recommenders.itemSim import ItemSim
from scripts.recommenders.weightedSim import WeightedSim
from scripts.recommenders.meanSim import MeanSim


RECOMMENDERS_TABLE = pd.DataFrame(
    [[1,  'ALS',           'ALS',      ALS,          ALS_HYPERPARAMETERS,  ALS_HYPERPARAMETERS],
     [2,  'BPR',           'BPR',      BPR,          BPR_HYPERPARAMETERS,  BPR_HYPERPARAMETERS],
     [3,  'ALS_itemSim',   'ALS',      ItemSim,      ALS_HYPERPARAMETERS,  ALS_ITEM_SIM_HYPERPARAMETERS],
     [4,  'BPR_itemSim',   'BPR',      ItemSim,      BPR_HYPERPARAMETERS,  BPR_ITEM_SIM_HYPERPARAMETERS],
     [5,  'ALS_weighted',  'ALS',      WeightedSim,  ALS_HYPERPARAMETERS,  ALS_WEIGHTED_SIM_HYPERPARAMETERS],
     [6,  'BPR_weighted',  'BPR',      WeightedSim,  BPR_HYPERPARAMETERS,  BPR_WEIGHTED_SIM_HYPERPARAMETERS],
     [7,  'ALS_mean',      'ALS',      MeanSim,      ALS_HYPERPARAMETERS,  ALS_MEAN_SIM_HYPERPARAMETERS],
     [8,  'BPR_mean',      'BPR',      MeanSim,      BPR_HYPERPARAMETERS,  BPR_MEAN_SIM_HYPERPARAMETERS]], 
    columns=[kw.RECOMMENDER_ID, kw.RECOMMENDER_NAME, kw.RECOMMENDER_EMBEDDINGS, kw.RECOMMENDER_CLASS, kw.EMBEDDINGS_HYPERPARAMETERS, kw.RECOMMENDER_HYPERPARAMETERS]
).set_index(kw.RECOMMENDER_ID)

class Recommender(object):
    def __init__(self, recommender_id):
        self.name = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_NAME]
        self.embeddings_name = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_EMBEDDINGS]
        self.model = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_CLASS]
        self.rec_hyperparameters = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_HYPERPARAMETERS]
        self.emb_hyperparameters = RECOMMENDERS_TABLE.loc[recommender_id, kw.EMBEDDINGS_HYPERPARAMETERS]

    def get_name(self):
        return self.name
    
    def get_embeddings_name(self):
        return self.embeddings_name
    
    def get_model(self):
        return self.model
    
    def get_recommender_hyperparameters(self):
        return self.rec_hyperparameters

    def get_embeddings_hyperparameters(self):
        return self.emb_hyperparameters
    
    def get_all_hyperparameters(self):
        return {
            **self.rec_hyperparameters,
            **self.emb_hyperparameters
        }
    
    def get_embeddings_hyperparameter_from_dict(self, all_hyperparameters_dict):
        emb_hyperparameters_dict = {}
        for key, value in all_hyperparameters_dict.items():
            if key in self.emb_hyperparameters:
                emb_hyperparameters_dict[key] = value
        return emb_hyperparameters_dict


def get_recommenders(recommenders=None):
    for recommender_id, recommender_data in RECOMMENDERS_TABLE.iterrows():
        if recommenders is None or recommender_data[kw.RECOMMENDER_NAME] in recommenders:
            yield Recommender(recommender_id)