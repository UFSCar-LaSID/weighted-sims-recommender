# Link para download da base original: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
import csv
from datetime import datetime
import os
import pandas as pd
import src as kw

def preprocess_retailrocket():
    print('Preprocessing RetailRocket...')
    # Pasta de dataset
    DATASET_FOLDER = 'retailrocket'

    # Nomes das colunas
    COLUMN_INTERACTION_TYPE = 'type'
    COLUMN_TRANSACTION_ID = 'id_transaction'
    COLUMN_DATETIME = 'datetime'
    COLUMN_TIMESTAMP = 'timestamp'

    # Define pasta de leitura
    intput_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)

    # ========================================= INTERAÇÕES =========================================
    # Lê CSV de interações
    df_interactions = pd.read_csv(os.path.join(intput_dir, 'events.csv'), sep=',', header=0, index_col=False)

    # Arruma nome das colunas
    df_interactions.columns = ['wrong_timestamp', kw.COLUMN_USER_ID, COLUMN_INTERACTION_TYPE, kw.COLUMN_ITEM_ID, COLUMN_TRANSACTION_ID]

    # Padroniza timestamps de itens da mesma interação
    last_timestamps = df_interactions.groupby(COLUMN_TRANSACTION_ID)['wrong_timestamp'].max()
    df_interactions[COLUMN_TIMESTAMP] = df_interactions[COLUMN_TRANSACTION_ID].map(last_timestamps.to_dict()).fillna(df_interactions['wrong_timestamp']).astype(int)
    df_interactions.drop(columns='wrong_timestamp')
    df_interactions = df_interactions.drop_duplicates(keep='first')

    # Modifica o timestamp para segundos ao invés de milisegundos (padrão das outras bases de dados)
    df_interactions[COLUMN_TIMESTAMP] = df_interactions[COLUMN_TIMESTAMP] // 1000

    # Gera datetime
    df_interactions[COLUMN_DATETIME] = df_interactions[COLUMN_TIMESTAMP].apply(lambda x: str(datetime.fromtimestamp(x)).split('.')[0])

    # Reordena colunas
    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, COLUMN_INTERACTION_TYPE, COLUMN_TRANSACTION_ID, COLUMN_TIMESTAMP, COLUMN_DATETIME]]
    df_interactions = df_interactions.sort_values([COLUMN_TIMESTAMP, kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID])


    # =========================================== ITENS ============================================
    # Lê CSVs de itens
    df_items_1 = pd.read_csv(os.path.join(intput_dir, 'item_properties_part1.csv'), sep=',', header=0, index_col=False)
    df_items_1 = df_items_1.drop_duplicates(keep='first')
    df_items_2 = pd.read_csv(os.path.join(intput_dir, 'item_properties_part2.csv'), sep=',', header=0, index_col=False)
    df_items_2 = df_items_2.drop_duplicates(keep='first')
    df_items = pd.concat([df_items_1, df_items_2], axis=0)

    # Arruma nome das colunas
    df_items.columns = [COLUMN_TIMESTAMP, kw.COLUMN_ITEM_ID, 'property', 'value']

    # Modifica o timestamp para segundos ao invés de milisegundos (padrão das outras bases de dados)
    df_items[COLUMN_TIMESTAMP] = df_items[COLUMN_TIMESTAMP] // 1000

    # Gera datetime
    df_items[COLUMN_DATETIME] = df_items[COLUMN_TIMESTAMP].apply(lambda x: str(datetime.fromtimestamp(x)).split('.')[0])

    # Reordena colunas
    df_items = df_items[[kw.COLUMN_ITEM_ID, 'property', 'value', COLUMN_TIMESTAMP, COLUMN_DATETIME]]
    df_items = df_items.sort_values([COLUMN_TIMESTAMP, kw.COLUMN_ITEM_ID])


    # =========================================== SALVAR ===========================================
    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Salva bases
    df_items.to_csv(os.path.join(output_dir, kw.FILE_ITEMS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)


    # ============================ GERAR ARQUIVOS APENAS COM TRANSACOES ============================
    df_interactions = df_interactions[df_interactions[COLUMN_INTERACTION_TYPE]=='transaction']
    df_interactions = df_interactions.drop(columns=[COLUMN_INTERACTION_TYPE])

    # =========================================== SALVAR ===========================================
    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, '{}-transactions'.format(DATASET_FOLDER))
    os.makedirs(output_dir, exist_ok=True)

    # Salva bases
    df_items.to_csv(os.path.join(output_dir, kw.FILE_ITEMS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)
    print('OK!')
