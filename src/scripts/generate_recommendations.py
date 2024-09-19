import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_path)

from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm

import src as kw
from src.dataset import get_datasets
from src.file_handlers import get_embeddings_filepath, log_recommendations
from src.recommenders import get_recommenders
from src.recsys import remove_single_interactions, remove_cold_start
from src.recommenders.mf import ALS, BPR
from src.parameters_handle import get_algo_and_dataset_parameters


recommenders_options, dataset_options = get_algo_and_dataset_parameters('Choose datasets and recommenders to train and generate recommendations')


for dataset in get_datasets(datasets=dataset_options):
    dataset_name = dataset.get_name()
    print('Loading dataset {}...'.format(dataset_name))
    
    df = dataset.get_dataframe()
    df = remove_single_interactions(df)
    
    kf = KFold(n_splits=kw.K_FOLD_SPLITS, shuffle=True, random_state=kw.RANDOM_STATE)
    for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
        df_train = df.iloc[train_index].copy()
        df_test = remove_cold_start(df_train, df.iloc[test_index].copy())
        
        for recommender in get_recommenders(recommenders=recommenders_options):
            recommender_name = recommender.get_name()
            print('Dataset: {} | Fold: {} | Recommender: {}'.format(dataset_name, fold, recommender_name))            
                        
            for parameters in tqdm(ParameterGrid(recommender.get_all_hyperparameters())):

                parameters['embeddings_filepath'] = get_embeddings_filepath(
                    dataset_name, 
                    recommender.get_embeddings_name(), 
                    recommender.get_embeddings_hyperparameter_from_dict(parameters), 
                    fold
                )
                
                if recommender.has_embeddings() and recommender.get_model() != recommender.get_embeddings_model():
                    if not os.listdir(parameters['embeddings_filepath']):
                        embeddings_parameters = {parameter_name: parameters[parameter_name] for parameter_name in recommender.get_embeddings_hyperparameters().keys()}
                        embeddings_parameters['embeddings_filepath'] = parameters['embeddings_filepath']

                        embeddings_model = recommender.get_embeddings_model()(**embeddings_parameters)
                        embeddings_model.fit(df_train)
                
                
                Model = recommender.get_model()
                model = Model(**parameters)
                model.fit(df_train)
                recommendations = model.recommend(df_test)

                if 'embeddings_filepath' in parameters.keys():
                    del parameters['embeddings_filepath']  # Para a nomenclatura correta final, é preciso remover isso da lista de params.
                
                rec_dir = log_recommendations(dataset_name, recommender_name, parameters, fold, df_test, recommendations)