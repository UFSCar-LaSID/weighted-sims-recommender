# Link para download da base original: https://grouplens.org/datasets/movielens/
from datetime import datetime
import os
import pandas as pd
import src as kw

def preprocess_ml1m():
    print('Preprocessing MovieLens-1M...')
    # Pasta de dataset
    DATASET_FOLDER = 'MovieLens-1M'
    COLUMN_TIMESTAMP = 'timestamp'

    # Define pasta de leitura
    intput_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)

    # ======================================= INTERACOES =======================================
    # Lê CSV de interações
    df_interactions = pd.read_csv(os.path.join(intput_dir, 'ratings.dat'), sep='::', engine='python', header=None)
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING, COLUMN_TIMESTAMP]
    # Gera datetimes

    # ========================================= SALVAR =========================================
    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Salva base
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)
    
    print('OK!')