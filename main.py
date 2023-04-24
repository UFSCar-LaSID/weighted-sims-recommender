from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm

from scripts.dataset import get_datasets
from scripts.file_handlers import get_embeddings_filepath, log_recommendations
from scripts.recommender import get_recommenders
from scripts.recsys import remove_single_interactions, remove_cold_start

K_SPLITS = 5
RANDOM_SEED = 1420

DATASETS = ['RetailRocket-Transactions']
RECOMMENDERS = ['ALS']

for dataset in get_datasets(datasets=DATASETS):
    dataset_name = dataset.get_name()
    print('Loading dataset {}...'.format(dataset_name))
    df = dataset.get_dataframe()
    df = remove_single_interactions(df)
    kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
        df_train = df.iloc[train_index]
        df_test = remove_cold_start(df.iloc[test_index])
        for Recommender, param_grid in get_recommenders(recommenders=RECOMMENDERS):
            recommender_name = Recommender.__name__
            print('Dataset: {} | Fold: {} | Recommender: {}'.format(dataset_name, fold, recommender_name))            
            for parameters in tqdm(ParameterGrid(param_grid)):
                embeddings_filepath = get_embeddings_filepath(dataset_name, recommender_name, parameters, fold)
                recommender = Recommender(embeddings_filepath=embeddings_filepath, random_seed=RANDOM_SEED, **parameters)
                recommender.fit(df_train)
                recommendations = recommender.recommend(df_test)
                log_recommendations(dataset_name, recommender_name, parameters, fold, df_test, recommendations)
