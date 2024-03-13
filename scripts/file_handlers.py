import os
import pandas as pd

import scripts as kw

MAIN_FOLDER = 'results'

def _dict_to_str(dictionary):
    return '_'.join(['{}-{}'.format(k,v) for k, v in sorted(dictionary.items())])

def get_embeddings_filepath(dataset_name, recommender_name, parameters, fold):
    parameters_string = _dict_to_str(parameters)
    filepath = os.path.join(MAIN_FOLDER, 'embeddings', dataset_name, recommender_name, parameters_string, str(fold))
    os.makedirs(filepath, exist_ok=True)
    return filepath

def get_recomendation_filepath(dataset_name, recommender_name, parameters):
    parameters_string = _dict_to_str(parameters)
    filepath = os.path.join(MAIN_FOLDER, 'recommendations', dataset_name, recommender_name, parameters_string)
    os.makedirs(filepath, exist_ok=True)
    return filepath

def log_recommendations(dataset_name, recommender_name, parameters, fold, df_test, recommendations):
    parameters_string = _dict_to_str(parameters)
    filedir = os.path.join(MAIN_FOLDER, 'recommendations', dataset_name, recommender_name, parameters_string, str(fold))
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, 'recommendations.csv')
    user_items = df_test.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].apply(lambda x: list(x))
    user_recs = recommendations.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].apply(lambda x: list(x))
    recommendations_match = pd.concat([user_items, user_recs], axis=1).reset_index()
    recommendations_match.columns = [kw.LOG_COLUMN_USER, kw.LOG_COLUMN_ITEMS, kw.LOG_COLUMN_RECOMMENDATIONS]
    recommendations_match.to_csv(filepath, sep=kw.DELIMITER, header=True, index=False, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR)
    return filedir

def log_items_similarity(file_path, items_similarities):
    items_similarities.to_csv(f'{file_path}/similarities.csv', sep=kw.DELIMITER, header=True, index=False, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR)