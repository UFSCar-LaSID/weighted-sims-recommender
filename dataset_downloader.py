import os

from scripts import FILE_INTERACTIONS
from scripts.dataset import download_datasets

DATASET_DIR = 'datasets'

datasets = download_datasets(ds_dir=DATASET_DIR)

for dataset in datasets:
    dir_path = os.path.join(DATASET_DIR, dataset)
    explicit_interactions = os.path.join(dir_path, 'interactions_explicit.csv')
    implicit_interactions = os.path.join(dir_path, 'interactions_implicit.csv')
    recurrent_interactions = os.path.join(dir_path, 'interactions_recurrent.csv')
    final_interactions = os.path.join(dir_path, FILE_INTERACTIONS)

    if os.path.exists(explicit_interactions):
        os.remove(explicit_interactions)

    if os.path.exists(recurrent_interactions):
        os.remove(recurrent_interactions)

    if os.path.exists(implicit_interactions):
        os.rename(implicit_interactions, final_interactions)