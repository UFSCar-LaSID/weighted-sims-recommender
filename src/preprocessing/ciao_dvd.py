

import pandas as pd
import os
import src as kw


def preprocess_ciaodvd():
    print('Preprocessing CiaoDVD...')

    COLUMN_DATETIME = 'datetime'
    DATASET_FOLDER = 'CiaoDVD'

    # Define pasta de leitura
    input_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)


    # Cria arquivo de interações
    print("Gerando arquivos de interação...")
    df_interactions = pd.read_csv(os.path.join(input_dir, "movie-ratings.txt"), header=None)
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, 'genreID', 'reviewID', kw.COLUMN_RATING, COLUMN_DATETIME]
    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING, COLUMN_DATETIME]]
    
    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Salva bases
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)
    print('OK!')