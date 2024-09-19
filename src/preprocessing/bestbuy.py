# Link para download da base original: https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big
from datetime import datetime
import os
import pandas as pd
import src as kw


def preprocess_bestbuy():
    print('Preprocessing BestBuy..')
    # Pasta de dataset
    DATASET_FOLDER = 'BestBuy' # Renomear a pasta para esse nome após o download

    # Nomes das colunas
    COLUMN_DATETIME = 'datetime'
    COLUMN_QUERY = 'query'
    COLUMN_QUERY_DATETIME = 'query_datetime'


    # Monta caminho da pasta de saída
    intput_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Le arquivo de treinamento
    df_interactions = pd.read_csv(os.path.join(intput_dir, 'train.csv'), sep=',', header=0)

    # Limpa e arruma arquivo de interações
    df_interactions = df_interactions.drop(columns='category')
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, COLUMN_QUERY, COLUMN_DATETIME, COLUMN_QUERY_DATETIME]
    # Padroniza formato dos datetimes
    df_interactions[COLUMN_DATETIME] = df_interactions[COLUMN_DATETIME].apply(lambda x: datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))

    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, COLUMN_DATETIME]]

    # Salva arquivos
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)
    print('OK!')