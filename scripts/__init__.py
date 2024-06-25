import csv
import numpy as np

# General configs for the experiments
RANDOM_STATE = 1420
K_FOLD_SPLITS = 5
TRAIN_MODE = 'cpu'
TOP_N = 20
K = 100
N_EVAL = 20

# Name of files of the processed datasets
FILE_ITEMS = 'items.csv'
FILE_USERS = 'users.csv'
FILE_INTERACTIONS = 'interactions.csv'

# Name of files for saving the models
FILE_SPARSE_REPR = 'sparse_repr.pkl'
FILE_ITEMS_EMBEDDINGS = 'items.npy'
FILE_USERS_EMBEDDINGS = 'users.npy'

# Column names for the datasets table
DATASET_ID = 'id'
DATASET_NAME = 'name'
DATASET_TYPE = 'type'
DATASET_SAMPLING_RATE = 'sampling_rate'

# Column names for the models table
RECOMMENDER_ID = 'id'
RECOMMENDER_NAME = 'name'
RECOMMENDER_EMBEDDINGS = 'embeddings'
RECOMMENDER_CLASS = 'class'
RECOMMENDER_HYPERPARAMETERS = 'specific_recommender_hyperparameter'
EMBEDDINGS_HYPERPARAMETERS = 'embeddings_hyperparameters'

# CSV configs
DELIMITER = ';'
QUOTECHAR = '"'
QUOTING = csv.QUOTE_ALL
ENCODING = "ISO-8859-1"

# Column names for the datasets CSVs
COLUMN_ITEM_ID = 'id_item'
COLUMN_ITEM_NAME = 'name_item'
COLUMN_USER_ID = 'id_user'
COLUMN_USER_NAME = 'name_user'
COLUMN_RATING = 'interaction'
COLUMN_RANK = 'rank'

# Column names for the log files
LOG_COLUMN_USER = 'user'
LOG_COLUMN_ITEMS = 'items'
LOG_COLUMN_RECOMMENDATIONS = 'recommendations'

# Memory size limit
MEM_SIZE_LIMIT = 3.2e+9