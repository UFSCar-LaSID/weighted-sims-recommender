
import pandas as pd
import os
import src as kw

def preprocess_anime_recommendations():
    # Pasta de dataset
    DATASET_FOLDER = 'AnimeRecommendations'

    # Define pasta de leitura
    input_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)


    # Cria arquivo de interações
    df_interactions = pd.read_csv(os.path.join(input_dir, 'rating.csv'), header=0, delimiter = ",")
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING]

    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)