
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_path)

import pandas as pd
import src as kw


from src.preprocessing.retailrocket import preprocess_retailrocket
from src.preprocessing.ml_1m import preprocess_ml1m
from src.preprocessing.delicious import preprocess_delicious
from src.preprocessing.bestbuy import preprocess_bestbuy
from src.preprocessing.film_trust import preprocess_filmtrust
from src.preprocessing.ciao_dvd import preprocess_ciaodvd
from src.preprocessing.last_fm import preprocess_last_fm
from src.preprocessing.anime import preprocess_anime_recommendations
from src.preprocessing.jester import preprocess_jester

from src.parameters_handle import get_input



# ----- Preprocess table -----

PREPROCESS_FUNCTION_NAME = 'preprocess_function'

PREPROCESS_TABLE = pd.DataFrame(
    [[1,  'AnimeRecommendations',   preprocess_anime_recommendations],
     [2,  'BestBuy',                preprocess_bestbuy],
     [3,  'CiaoDVD',                preprocess_ciaodvd],
     [4,  'DeliciousBookmarks',     preprocess_delicious],
     [5,  'FilmTrust',              preprocess_filmtrust],
     [6,  'Jester',                 preprocess_jester],
     [7,  'LastFM',                 preprocess_last_fm],
     [8,  'MovieLens-1M',           preprocess_ml1m],
     [9,  'RetailRocket',           preprocess_retailrocket]],
    columns=[kw.DATASET_ID, kw.DATASET_NAME, PREPROCESS_FUNCTION_NAME]
).set_index(kw.DATASET_ID)





# ----- Collect dataset options from command line or interactive menu -----

options = get_input('Choose datasets to preprocess', [
    {
        'name': 'datasets',
        'description': 'Dataset names (or indexes) to preprocess. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be preprocessed.',
        'options': PREPROCESS_TABLE,
        'name_column': kw.DATASET_NAME
    }
])[0]


# ----- Preprocess datasets -----

for option_index in options:
    dataset_name = PREPROCESS_TABLE.loc[option_index, kw.DATASET_NAME]
    print('Preprocessing {}...'.format(dataset_name))
    preprocess_function = PREPROCESS_TABLE.loc[option_index, PREPROCESS_FUNCTION_NAME]
    preprocess_function()
    print('Preprocessing {} done!'.format(dataset_name))