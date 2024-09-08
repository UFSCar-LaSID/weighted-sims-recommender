# Link para download da base original: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
import csv
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm

# Pasta de dataset
DATASET_FOLDER = 'netflixprize' # Renomear a pasta para esse nome após o download
RAW_FOLDER = 'raw' # Salvar os dados dentro dessa pasta após o download
PROCESSED_FOLDER = 'processed'

# Nomes de arquivos
FILE_ITEMS = 'items.csv'
FILE_INTERACTIONS = 'interactions.csv'

# Dados dos CSVs
DELIMITER = ';'
ENCODING = "utf-8"
QUOTING = csv.QUOTE_ALL
QUOTECHAR = '"'

# Nomes das colunas
COLUMN_ITEM_ID = 'id_item'
COLUMN_USER_ID = 'id_user'
COLUMN_RATING = 'rating'
COLUMN_DATETIME = 'datetime'
COLUMN_TIMESTAMP = 'timestamp'
COLUMN_TITLE = 'title'
COLUMN_RELEASE_YEAR = 'release_year'

# Define pasta onde vai salvar
intput_dir = os.path.join(RAW_FOLDER, DATASET_FOLDER)
output_dir = os.path.join(PROCESSED_FOLDER, DATASET_FOLDER)
os.makedirs(output_dir, exist_ok=True)

# Inicializa dataframe de interações
df_interactions = pd.DataFrame()

# Processa cada arquivo 
for file_id in range(1, 5):
    file_name = 'combined_data_{}.txt'.format(file_id)
    print('Processing file {}...'.format(file_name))
    item_interactions = list()
    for line in tqdm(open(os.path.join(intput_dir, file_name), 'r').readlines()):        
        if ':' in line:
            # Converte interacoes para dataframe
            if len(item_interactions) != 0:
                item_df = pd.DataFrame(item_interactions, columns=[COLUMN_USER_ID, COLUMN_RATING, COLUMN_DATETIME])
                item_df[COLUMN_ITEM_ID] = item_id
                item_df[COLUMN_DATETIME] += ' 00:00:00'
                item_df[COLUMN_DATETIME] = item_df[COLUMN_DATETIME].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                item_df[COLUMN_TIMESTAMP] = item_df[COLUMN_DATETIME].apply(lambda x: int(datetime.timestamp(x)))
                # Adiciona dataframe de interações do item no dataframe geral
                df_interactions = pd.concat([df_interactions, item_df], axis=0)
            # Captura o novo item
            item_id = int(line.split(':')[0])
            item_interactions = list()
        else:
            # Captura a interação
            item_interactions.append(line.strip().split(','))
        
    # Arruma ordem das colunas
    df_interactions = df_interactions[[COLUMN_USER_ID, COLUMN_ITEM_ID, COLUMN_RATING, COLUMN_DATETIME, COLUMN_TIMESTAMP]].sort_values([COLUMN_TIMESTAMP, COLUMN_USER_ID, COLUMN_ITEM_ID])

    # Salva arquivo
    print('Saving file {}...'.format(file_name))
    df_interactions.to_csv(
        os.path.join(output_dir, FILE_INTERACTIONS), 
        sep=DELIMITER, encoding=ENCODING, quoting=QUOTING, quotechar=QUOTECHAR, 
        header=(True if file_id == 1 else False), index=False, mode='a'
    )

    # Reseta dataframe de interações
    df_interactions = pd.DataFrame()

# Arruma e salva arquivo de filmes
df_items = list()
for line in tqdm(open(os.path.join(intput_dir, 'movie_titles.csv'), 'r', encoding='latin3').readlines()):
    ls = line.split(',')                
    df_items.append([ls[0].strip(), ls[1].strip(), ','.join(ls[2:]).strip()])
df_items = pd.DataFrame(df_items)
df_items.columns = [COLUMN_ITEM_ID, COLUMN_RELEASE_YEAR, COLUMN_TITLE]
df_items[COLUMN_RELEASE_YEAR] = df_items[COLUMN_RELEASE_YEAR].str.replace('NULL', '')
df_items.to_csv(os.path.join(output_dir, FILE_ITEMS), sep=DELIMITER, encoding=ENCODING, quoting=QUOTING, quotechar=QUOTECHAR, header=True, index=False)
print('OK')