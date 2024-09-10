import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_path)

import src as kw
from src.dataset import get_datasets
from src.file_handlers import get_recomendation_filepath
from src.recommenders import get_recommenders
from src.metrics import Metrics
from src.parameters_handle import get_algo_and_dataset_parameters


recommenders_options, dataset_options = get_algo_and_dataset_parameters('Choose datasets and recommenders to train and generate recommendations')


for dataset in get_datasets(datasets=dataset_options):
    dataset_name = dataset.get_name()

    for recommender in get_recommenders(recommenders=recommenders_options):
        recommender_name = recommender.get_name()

        print('Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

        model = Metrics(kw.K_FOLD_SPLITS, kw.N_EVAL)

        recomendation_filepath = get_recomendation_filepath(dataset_name, recommender_name)

        model.add_metrics(recomendation_filepath)

        save_path = model.save_metrics(dataset_name, recommender_name)

        print('The best results for {} and {} are:'.format(dataset_name, recommender_name))
        model.print_best_results()

        print('Metrics saved for {} and {}'.format(dataset_name, recommender_name))
        print('Saved at: {}\n\n'.format(save_path))
