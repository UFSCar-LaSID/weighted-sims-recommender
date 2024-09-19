import pandas as pd

import src as kw
from src.recommenders.mf import ALS, BPR
from src.recommenders.hyperparameters import ALS_HYPERPARAMETERS, BPR_HYPERPARAMETERS, ALS_ITEM_SIM_HYPERPARAMETERS, BPR_ITEM_SIM_HYPERPARAMETERS, ALS_WEIGHTED_SIM_HYPERPARAMETERS, BPR_WEIGHTED_SIM_HYPERPARAMETERS, REC_VAE_HYPERPARAMETERS, REC_VAE_ITEM_SIM_HYPERPARAMETERS, REC_VAE_WEIGHTED_HYPERPARAMETERS
from src.recommenders.itemSim import ItemSim
from src.recommenders.weightedSim import WeightedSim
from src.recommenders.meanSim import MeanSim
from src.recommenders.RecVAE.RecVAE import RecVAE


RECOMMENDERS_TABLE = pd.DataFrame(
    [[1,  'ALS',              'ALS',      ALS,          ALS_HYPERPARAMETERS,      ALS_HYPERPARAMETERS],
     [2,  'BPR',              'BPR',      BPR,          BPR_HYPERPARAMETERS,      BPR_HYPERPARAMETERS],
     [3,  'RecVAE',           'RecVAE',   RecVAE,       REC_VAE_HYPERPARAMETERS,  REC_VAE_HYPERPARAMETERS],
     [4,  'ALS_itemSim',      'ALS',      ItemSim,      ALS_HYPERPARAMETERS,      ALS_ITEM_SIM_HYPERPARAMETERS],
     [5,  'BPR_itemSim',      'BPR',      ItemSim,      BPR_HYPERPARAMETERS,      BPR_ITEM_SIM_HYPERPARAMETERS],
     [6,  'RecVAE_itemSim',   'RecVAE',   ItemSim,      REC_VAE_HYPERPARAMETERS,  REC_VAE_ITEM_SIM_HYPERPARAMETERS],
     [7,  'ALS_weighted',     'ALS',      WeightedSim,  ALS_HYPERPARAMETERS,      ALS_WEIGHTED_SIM_HYPERPARAMETERS],
     [8,  'BPR_weighted',     'BPR',      WeightedSim,  BPR_HYPERPARAMETERS,      BPR_WEIGHTED_SIM_HYPERPARAMETERS],
     [9,  'RecVAE_weighted',  'RecVAE',   WeightedSim,  REC_VAE_HYPERPARAMETERS,  REC_VAE_WEIGHTED_HYPERPARAMETERS]], 
    columns=[kw.RECOMMENDER_ID, kw.RECOMMENDER_NAME, kw.RECOMMENDER_EMBEDDINGS, kw.RECOMMENDER_CLASS, kw.EMBEDDINGS_HYPERPARAMETERS, kw.RECOMMENDER_HYPERPARAMETERS]
).set_index(kw.RECOMMENDER_ID)

embeddings_names_to_models = {
    'ALS': ALS,
    'BPR': BPR,
    'RecVAE': RecVAE
}

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
    
    def get_embeddings_model(self):
        return embeddings_names_to_models[self.get_embeddings_name()]
    
    def get_model(self):
        return self.model
    
    def has_embeddings(self):
        return self.embeddings_name != ''
    
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
        if recommender_id in recommenders:
            yield Recommender(recommender_id)