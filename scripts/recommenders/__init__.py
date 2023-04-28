import pandas as pd

import scripts as kw
from scripts.recommenders.mf import ALS, BPR
from scripts.recommenders.hyperparameters import ALS_HYPERPARAMETERS, BPR_HYPERPARAMETERS


RECOMMENDERS_TABLE = pd.DataFrame(
    [[1,  'ALS',    ALS,    ALS_HYPERPARAMETERS],
     [2,  'BPR',    BPR,    BPR_HYPERPARAMETERS]], 
    columns=[kw.RECOMMENDER_ID, kw.RECOMMENDER_NAME, kw.RECOMMENDER_CLASS, kw.RECOMMENDER_HYPERPARAMETERS]
).set_index(kw.RECOMMENDER_ID)


def get_recommenders(recommenders=None):
    for _, recommender_data in RECOMMENDERS_TABLE.iterrows():
        if recommenders is None or recommender_data[kw.RECOMMENDER_NAME] in recommenders:
            yield recommender_data[kw.RECOMMENDER_CLASS], recommender_data[kw.RECOMMENDER_HYPERPARAMETERS]