from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm
import os

import src as kw
from src.dataset import get_datasets
from src.file_handlers import get_embeddings_filepath, get_recomendation_filepath, log_recommendations
from src.recommenders import get_recommenders
from src.recsys import remove_single_interactions, remove_cold_start
from src.recommenders.mf import ALS, BPR
from src.metrics import Metrics

DATASETS = [
    'RetailRocket-Transactions',
    #'DeliciousBookmarks', 'BestBuy', 'Book-Crossing', 'Jester', 
    #'Anime Recommendations', 'MovieLens', 'NetflixPrize', 'LibimSeTi'
]
RECOMMENDERS = ['ALS_weighted', 'BPR_weighted'] # Mudar recomendadores aqui ALS_weighted ALS_mean
MODES = ['Evaluate'] # Mudar o comportamento do programa aqui

for MODE in MODES:

    if (MODE == 'Recommend'):

        for dataset in get_datasets(datasets=DATASETS):
            dataset_name = dataset.get_name()
            print('Loading dataset {}...'.format(dataset_name))
            
            df = dataset.get_dataframe()
            df = remove_single_interactions(df)
            
            kf = KFold(n_splits=kw.K_FOLD_SPLITS, shuffle=True, random_state=kw.RANDOM_STATE)
            for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
                
                df_train = df.iloc[train_index].copy()
                df_test = remove_cold_start(df_train, df.iloc[test_index].copy())
                
                for recommender in get_recommenders(recommenders=RECOMMENDERS):
                    recommender_name = recommender.get_name()
                    print('Dataset: {} | Fold: {} | Recommender: {}'.format(dataset_name, fold, recommender_name))            
                                
                    for parameters in tqdm(ParameterGrid(recommender.get_all_hyperparameters())):

                        embeddings_filepath = get_embeddings_filepath(
                            dataset_name, 
                            recommender.get_embeddings_name(), 
                            recommender.get_embeddings_hyperparameter_from_dict(parameters), 
                            fold
                        )

                        if not os.listdir(embeddings_filepath):
                            if (recommender.get_embeddings_name() == "ALS"):
                                embedding_model = ALS(embeddings_filepath, **parameters)
                                embedding_model.fit(df_train)
                            elif (recommender.get_embeddings_name() == "BPR"):
                                embedding_model = BPR(embeddings_filepath, **parameters)
                                embedding_model.fit(df_train)
                            else :
                                raise Exception("Invalid embedding model")
                        
                        Model = recommender.get_model()
                        model = Model(embeddings_filepath=embeddings_filepath, **parameters)
                        model.fit(df_train)
                        recommendations = model.recommend(df_test)

                        rec_dir = log_recommendations(dataset_name, recommender_name, parameters, fold, df_test, recommendations)

    if (MODE == 'Evaluate'):

        for dataset in get_datasets(datasets=DATASETS):
            dataset_name = dataset.get_name()

            for recommender in get_recommenders(recommenders=RECOMMENDERS):
                recommender_name = recommender.get_name()

                print('Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

                model = Metrics(kw.K_FOLD_SPLITS, kw.N_EVAL)

                recomendation_filepath = get_recomendation_filepath(dataset_name, recommender_name)

                model.add_metrics(recomendation_filepath)

                model.save_metrics(dataset_name, recommender_name)
