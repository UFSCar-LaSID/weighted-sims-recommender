import csv
import os
import pandas as pd
import src as kw

def preprocess_jester():
    CSV_FORMAT_CSV = {
        'delimiter': kw.DELIMITER,
        'quotechar': kw.QUOTECHAR,
        'quoting': kw.QUOTING
    }
    # Cria a pasta para download
    DATASET_FOLDER = 'Jester'

    # Define pasta de leitura
    input_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)

    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, DATASET_FOLDER)
    os.makedirs(output_dir, exist_ok=True)



    # Processa arquivo de interacoes
    n_users = 0
    with open(os.path.join(output_dir, kw.FILE_INTERACTIONS), 'w', encoding='utf-8') as explicit_fout:
        explicit_csv_writer = csv.writer(explicit_fout, **CSV_FORMAT_CSV)
        explicit_csv_writer.writerow([kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING])
        for dataset in ['jester-data-1.xls', 'jester-data-2.xls', 'jester-data-3.xls', 'jesterfinal151cols.xls']:
            ratings = pd.read_excel(os.path.join(input_dir, dataset), header=None, index_col=None).drop(columns=[0])
            for i, (_, user_ratings) in enumerate(ratings.iterrows()):
                user_id = n_users + i
                for item_id, rating in user_ratings[user_ratings != 99].items(): # Limpa avaliacoes nulas
                    explicit_csv_writer.writerow([user_id, item_id, rating])
            n_users += ratings.shape[0]
