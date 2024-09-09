
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_path)

import argparse
import pandas as pd
import src as kw
from src.preprocessing.retailrocket import preprocess_retailrocket
from src.parameters_handle import get_input



# ----- Preprocess table -----

PREPROCESS_FUNCTION_NAME = 'preprocess_function'

PREPROCESS_TABLE = pd.DataFrame(
    [[1,  'RetailRocket', preprocess_retailrocket]], 
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
    preprocess_function = PREPROCESS_TABLE.loc[option_index, PREPROCESS_FUNCTION_NAME]
    preprocess_function()
    print('Preprocessing {} done!'.format(dataset_name))