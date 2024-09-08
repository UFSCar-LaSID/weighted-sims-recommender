
import argparse
import pandas as pd
import scripts as kw
from scripts.preprocessing.retailrocket import preprocess_retailrocket
from scripts.parameters_handle import ask_options



# ----- Preprocess table -----

PREPROCESS_FUNCTION_NAME = 'preprocess_function'

PREPROCESS_TABLE = pd.DataFrame(
    [[1,  'RetailRocket', preprocess_retailrocket]], 
    columns=[kw.DATASET_ID, kw.DATASET_NAME, PREPROCESS_FUNCTION_NAME]
).set_index(kw.DATASET_ID)





# ----- Collect dataset options from command line or interactive menu -----

parser = argparse.ArgumentParser(description='Framework de avaliação 3D genérico.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datasets', type=str, default=None, help='Dataset names (or indexes) to preprocess. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be preprocessed.')

args = parser.parse_args()

dataset_option = args.datasets

if dataset_option is None:
    options = ask_options('dataset', PREPROCESS_TABLE, kw.DATASET_NAME)
elif dataset_option == 'all':
    options = PREPROCESS_TABLE.index.tolist()
elif dataset_option.replace(" ", "").isdigit():
    options = []
    dataset_option = dataset_option.split(' ')
    for option in dataset_option:
        if int(option) not in PREPROCESS_TABLE.index:
            raise ValueError('Dataset index {} not found!'.format(option))
        options.append(int(option))
else:
    options = []
    options_names = dataset_option.split(' ')
    for option_name in options_names:
        if not PREPROCESS_TABLE[kw.DATASET_NAME].str.contains(option_name).any():
            raise ValueError('Dataset {} not found!'.format(option_name))
        options.append(PREPROCESS_TABLE[PREPROCESS_TABLE[kw.DATASET_NAME].str.contains(option_name)].index.tolist()[0])

options = list(set(options))




# ----- Preprocess datasets -----

for option_index in options:
    dataset_name = PREPROCESS_TABLE.loc[option_index, kw.DATASET_NAME]
    preprocess_function = PREPROCESS_TABLE.loc[option_index, PREPROCESS_FUNCTION_NAME]
    preprocess_function()
    print('Preprocessing {} done!'.format(dataset_name))