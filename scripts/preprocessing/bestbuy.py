# Link para download da base original: https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big
import csv
from datetime import datetime
import os
import pandas as pd
import shutil
from sklearn.preprocessing import LabelEncoder

# Pasta de dataset
DATASET_FOLDER = 'bestbuy' # Renomear a pasta para esse nome após o download
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
COLUMN_DATETIME = 'datetime'
COLUMN_TIMESTAMP = 'timestamp'
COLUMN_CATEGORY = 'category'
COLUMN_QUERY = 'query'
COLUMN_QUERY_DATETIME = 'query_datetime'
COLUMN_QUERY_TIMESTAMP = 'query_timestamp'

# Monta caminho da pasta de saída
intput_dir = os.path.join(RAW_FOLDER, DATASET_FOLDER)
output_dir = os.path.join(PROCESSED_FOLDER, DATASET_FOLDER)
os.makedirs(output_dir, exist_ok=True)

# Copia dados dos itens
shutil.copy(os.path.join(intput_dir, 'product_data.tar.gz'), os.path.join(output_dir, 'product_data.tar.gz'))

# Le arquivo de treinamento
df_interactions = pd.read_csv(os.path.join(intput_dir, 'train.csv'), sep=',', header=0)

# Gera arquivo de itens
df_items = df_interactions[['sku', 'category']].drop_duplicates(keep='first').copy()
df_items.columns = [COLUMN_ITEM_ID, COLUMN_CATEGORY]

# Limpa e arruma arquivo de interações
df_interactions = df_interactions.drop(columns='category')
df_interactions.columns = [COLUMN_USER_ID, COLUMN_ITEM_ID, COLUMN_QUERY, COLUMN_DATETIME, COLUMN_QUERY_DATETIME]
# Padroniza formato dos datetimes
df_interactions[COLUMN_DATETIME] = df_interactions[COLUMN_DATETIME].apply(lambda x: datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
df_interactions[COLUMN_QUERY_DATETIME] = df_interactions[COLUMN_QUERY_DATETIME].apply(lambda x: datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
# Gera timestamps
df_interactions[COLUMN_TIMESTAMP] = df_interactions[COLUMN_DATETIME].apply(lambda x: int(datetime.timestamp(x)))
df_interactions[COLUMN_QUERY_TIMESTAMP] = df_interactions[COLUMN_QUERY_DATETIME].apply(lambda x: int(datetime.timestamp(x)))

# Salva arquivos
for df, file_name in [(df_interactions, FILE_INTERACTIONS), (df_items, FILE_ITEMS)]:
    df.to_csv(os.path.join(output_dir, file_name), sep=DELIMITER, encoding=ENCODING, quoting=QUOTING, quotechar=QUOTECHAR, header=True, index=False)
print('OK!')