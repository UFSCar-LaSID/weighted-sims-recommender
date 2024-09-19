

import pandas as pd
import os
import src as kw


def preprocess_last_fm():
    print('Preprocessing LastFM...')

    DATASET_FOLDER = 'LastFM'

    # Define pasta de leitura
    input_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)


    # Cria arquivo de interações
    print("Gerando arquivos de interação...")
    df_interactions = pd.read_csv(os.path.join(input_dir, "user_artists.dat"), header=0, sep='\t')
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, 'weight']
    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]]
    
    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Salva bases
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)
    print('OK!')