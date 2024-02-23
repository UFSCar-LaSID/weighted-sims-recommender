from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm
import os

import scripts as kw
from scripts.dataset import get_datasets
from scripts.file_handlers import get_embeddings_filepath, log_recommendations, log_items_similarity
from scripts.recommenders import get_recommenders
from scripts.recsys import remove_single_interactions, remove_cold_start
from scripts.recommenders.mf import ALS, BPR

DATASETS = ['RetailRocket-Transactions'] # Mudar bases de dados aqui
RECOMMENDERS = ['ALS_itemSim'] # Mudar recomendadores aqui

for dataset in get_datasets(datasets=DATASETS):
    dataset_name = dataset.get_name()
    print('Loading dataset {}...'.format(dataset_name))
    
    df = dataset.get_dataframe()
    df = remove_single_interactions(df)
    
    kf = KFold(n_splits=kw.K_FOLD_SPLITS, shuffle=True, random_state=kw.RANDOM_STATE)
    for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
        
        df_train = df.iloc[train_index]
        df_test = remove_cold_start(df_train, df.iloc[test_index])
        
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
                        embedding_model = ALS(embeddings_filepath)
                        embedding_model.fit(df_train)
                    elif (recommender.get_embeddings_name() == "BPR"):
                        embedding_model = BPR(embeddings_filepath)
                        embedding_model.fit(df_train)
                    else :
                        raise Exception("Invalid embedding model")
                
                Model = recommender.get_model()
                model = Model(embeddings_filepath=embeddings_filepath, **parameters)
                model.fit(df_train)
                recommendations = model.recommend(df_test)
                
                rec_dir = log_recommendations(dataset_name, recommender_name, parameters, fold, df_test, recommendations)

                # Se tiver o método get_items_sims, salvar as similaridades
                # has_items_sims = getattr(model, 'get_items_sims', None)
                # if callable(has_items_sims):
                    # log_items_similarity(rec_dir, model.get_items_sims())
