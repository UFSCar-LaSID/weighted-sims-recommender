import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta

DATASET_PATH = 'raw'
SAVE_PATH = 'processed'

# Nomes de arquivos
FILE_ITEMS = 'items.csv'
FILE_USERS = 'users.csv'
FILE_INTERACTIONS = 'interactions.csv'

# Colunas dos CSV e DataFrames
COLUMN_ITEM_ID = 'id_item'
COLUMN_USER_ID = 'id_user'
COLUMN_RATING = 'rating'
COLUMN_DATETIME = 'datetime'
COLUMN_TIMESTAMP = 'timestamp'
COLUMN_ITEM_NAME = 'name_item'
COLUMN_USER_NAME = 'name_user'
COLUMN_RANK = 'rank'

# Separador dos CSVs
CSV_SEP = ';'

items_df = pd.read_json(f'{DATASET_PATH}/yelp_academic_dataset_business.json', lines=True)
users_df = pd.read_json(f'{DATASET_PATH}/yelp_academic_dataset_user.json', lines=True)
interactions_df = pd.read_json(f'{DATASET_PATH}/yelp_academic_dataset_review.json', lines=True)

rename_dict_users = {
    'user_id': COLUMN_USER_ID,
    'name': COLUMN_USER_NAME
}
users_df.rename(columns=rename_dict_users, inplace=True)

rename_dict_items = {
    'business_id': COLUMN_ITEM_ID,
    'name': COLUMN_ITEM_NAME
}
items_df.rename(columns=rename_dict_items, inplace=True)

rename_dict_interactions = {
    'stars': COLUMN_RATING,
    'date': COLUMN_DATETIME,
    'user_id': COLUMN_USER_ID,
    'business_id': COLUMN_ITEM_ID
}
interactions_df.rename(columns=rename_dict_interactions, inplace=True)

interactions_df.drop_duplicates(subset=[COLUMN_ITEM_ID, COLUMN_USER_ID, COLUMN_DATETIME], inplace=True)
interactions_df[COLUMN_TIMESTAMP] = interactions_df[COLUMN_DATETIME].astype('datetime64[ns]').apply(lambda x: str(datetime.timestamp(x)).split('.')[0])

users_df.to_csv(f'{SAVE_PATH}/{FILE_USERS}', index=False, sep=CSV_SEP)
items_df.to_csv(f'{SAVE_PATH}/{FILE_ITEMS}', index=False, sep=CSV_SEP)
interactions_df.to_csv(f'{SAVE_PATH}/{FILE_INTERACTIONS}', index=False, sep=CSV_SEP)
