# Link para download da base original: https://grouplens.org/datasets/hetrec-2011/
import os
import pandas as pd
import src as kw

def preprocess_delicious():
    # Pasta de dataset
    DATASET_FOLDER = 'DeliciousBookmarks' # Renomear a pasta para esse nome após o download
    COLUMN_DATETIME = 'datetime'

    # Define pasta de leitura
    input_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)

    # ================================================================================================
    # ========================================= URL COMPLETA =========================================
    # ================================================================================================

    # ----------------------------------- INTERACTIONS & BOOKMARKS -----------------------------------
    # Lê CSV de tags
    df_interactions = pd.read_csv(os.path.join(input_dir, 'user_taggedbookmarks.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
    df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']] = df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']].astype(str)
    df_interactions[COLUMN_DATETIME] =  df_interactions['year'].str.zfill(4) + '-' + df_interactions['month'].str.zfill(2) + '-' + df_interactions['day'].str.zfill(2) + ' ' + df_interactions['hour'].str.zfill(2) + ':' + df_interactions['minute'].str.zfill(2) + ':' + df_interactions['second'].str.zfill(2)

    # Adiciona ID padronizado nas interações
    df_interactions[kw.COLUMN_ITEM_ID] = df_interactions['bookmarkID']
    
    # Limpa e formata os dataframes
    df_interactions = df_interactions.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'bookmarkID', 'tagID'])
    df_interactions.columns = [kw.COLUMN_USER_ID, COLUMN_DATETIME, kw.COLUMN_ITEM_ID]
    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, COLUMN_DATETIME]]
    df_interactions = df_interactions.drop_duplicates()

    # ------------------------------------------- SALVAR --------------------------------------------
    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Salva base
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)


