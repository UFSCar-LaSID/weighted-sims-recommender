# Link para download da base original: https://grouplens.org/datasets/hetrec-2011/
import csv
from datetime import datetime
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Pasta de dataset
DATASET_FOLDER = 'delicious2k' # Renomear a pasta para esse nome após o download
RAW_FOLDER = 'raw' # Salvar os dados dentro dessa pasta após o download
PROCESSED_FOLDER = 'processed'

# Nomes de arquivos
FILE_ITEMS = 'items.csv'
FILE_INTERACTIONS = 'interactions.csv'
FILE_CONTACTS = 'contacts.csv'
FILE_TAGS = 'tags.csv'

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
COLUMN_CONTACT_ID = 'id_contact'
COLUMN_TAG_ID = 'id_tag'
COLUMN_TAG_NAME = 'name_tag'
COLUMN_TITLE = 'title'
COLUMN_URL = 'url'
COLUMN_URL_PRINCIPAL = 'urlPrincipal'

# Define pasta de leitura
intput_dir = os.path.join(RAW_FOLDER, DATASET_FOLDER)

# ================================================================================================
# ========================================= URL COMPLETA =========================================
# ================================================================================================

# ----------------------------------- INTERACTIONS & BOOKMARKS -----------------------------------
# Lê CSV de tags
df_interactions = pd.read_csv(os.path.join(intput_dir, 'user_taggedbookmarks.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']] = df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']].astype(str)
df_interactions[COLUMN_DATETIME] =  df_interactions['year'].str.zfill(4) + '-' + df_interactions['month'].str.zfill(2) + '-' + df_interactions['day'].str.zfill(2) + ' ' + df_interactions['hour'].str.zfill(2) + ':' + df_interactions['minute'].str.zfill(2) + ':' + df_interactions['second'].str.zfill(2)

# Gera timestamp
df_interactions[COLUMN_TIMESTAMP] = df_interactions[COLUMN_DATETIME].apply(lambda x: int(datetime.timestamp(datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))))

# Lê CSV de bookmarks
df_bookmarks = pd.read_csv(os.path.join(intput_dir, 'bookmarks.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
df_bookmarks = df_bookmarks.drop(columns=['md5', 'md5Principal'])

# Padroniza URLs
le_url_lower = LabelEncoder()
df_bookmarks[COLUMN_ITEM_ID] = le_url_lower.fit_transform(df_bookmarks['url'].str.lower())
id_dict = df_bookmarks.set_index('id')[COLUMN_ITEM_ID].to_dict()

# Adiciona ID padronizado nas interações
df_interactions[COLUMN_ITEM_ID] = df_interactions['bookmarkID'].map(id_dict)

# Limpa e formata os dataframes
df_interactions = df_interactions.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'bookmarkID'])
df_interactions.columns = [COLUMN_USER_ID, COLUMN_TAG_ID, COLUMN_DATETIME, COLUMN_TIMESTAMP, COLUMN_ITEM_ID]
df_interactions = df_interactions[[COLUMN_USER_ID, COLUMN_ITEM_ID, COLUMN_TAG_ID, COLUMN_TIMESTAMP, COLUMN_DATETIME]].sort_values(COLUMN_DATETIME)

df_bookmarks = df_bookmarks.drop(columns=['id'])
df_bookmarks.columns = [COLUMN_TITLE, COLUMN_URL, COLUMN_URL_PRINCIPAL, COLUMN_ITEM_ID]
df_bookmarks = df_bookmarks[[COLUMN_ITEM_ID, COLUMN_TITLE, COLUMN_URL, COLUMN_URL_PRINCIPAL]].sort_values(COLUMN_ITEM_ID)

# ------------------------------------------ USUARIOS -------------------------------------------
# Lê CSV de usuários
df_users = pd.read_csv(os.path.join(intput_dir, 'user_contacts.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
df_users[['date_year', 'date_month', 'date_day', 'date_hour', 'date_minute', 'date_second']] = df_users[['date_year', 'date_month', 'date_day', 'date_hour', 'date_minute', 'date_second']].astype(str)
df_users[COLUMN_DATETIME] =  df_users['date_year'].str.zfill(4) + '-' + df_users['date_month'].str.zfill(2) + '-' + df_users['date_day'].str.zfill(2) + ' ' + df_users['date_hour'].str.zfill(2) + ':' + df_users['date_minute'].str.zfill(2) + ':' + df_users['date_second'].str.zfill(2)

# Gera timestamp
df_users[COLUMN_TIMESTAMP] = df_users[COLUMN_DATETIME].apply(lambda x: int(datetime.timestamp(datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))))

# Limpa o dataframe de usuários
df_users = df_users.drop(columns=['date_year', 'date_month', 'date_day', 'date_hour', 'date_minute', 'date_second'])
df_users.columns = [COLUMN_USER_ID, COLUMN_CONTACT_ID, COLUMN_DATETIME, COLUMN_TIMESTAMP]

# -------------------------------------------- TAGS ---------------------------------------------
# Lê CSV de tags
df_tags = pd.read_csv(os.path.join(intput_dir, 'tags.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
df_tags.columns = [COLUMN_TAG_ID, COLUMN_TAG_NAME]

# ------------------------------------------- SALVAR --------------------------------------------
# Cria pasta de saída
output_dir = os.path.join(PROCESSED_FOLDER, DATASET_FOLDER)
os.makedirs(output_dir, exist_ok=True)

# Salva bases
for df, file_name in [(df_interactions, FILE_INTERACTIONS), (df_bookmarks, FILE_ITEMS), (df_users, FILE_CONTACTS), (df_tags, FILE_TAGS)]:
    df.to_csv(os.path.join(output_dir, file_name), sep=DELIMITER, encoding=ENCODING, quoting=QUOTING, quotechar=QUOTECHAR, header=True, index=False)
print('OK!')


# ================================================================================================
# ======================================== URL PRINCIPAL =========================================
# ================================================================================================

# ----------------------------------- INTERACTIONS & BOOKMARKS -----------------------------------
# Gera um novo ID
le_url_principal = LabelEncoder()
df_bookmarks['urlPrincipalID'] = le_url_principal.fit_transform(df_bookmarks[COLUMN_URL_PRINCIPAL].str.lower())
id_dict = df_bookmarks.set_index(COLUMN_ITEM_ID)['urlPrincipalID'].to_dict()
df_interactions['urlPrincipalID'] = df_interactions[COLUMN_ITEM_ID].map(id_dict)

# Limpa dataframes
df_bookmarks = df_bookmarks.drop(columns=[COLUMN_ITEM_ID, COLUMN_TITLE, COLUMN_URL]).drop_duplicates(keep='first').rename(columns={'urlPrincipalID': COLUMN_ITEM_ID})[[COLUMN_ITEM_ID, COLUMN_URL_PRINCIPAL]].sort_values(COLUMN_ITEM_ID)
df_interactions = df_interactions.drop(columns=[COLUMN_ITEM_ID]).rename(columns={'urlPrincipalID': COLUMN_ITEM_ID})[[COLUMN_USER_ID, COLUMN_ITEM_ID, COLUMN_TAG_ID, COLUMN_TIMESTAMP, COLUMN_DATETIME]].sort_values(COLUMN_DATETIME)

# ------------------------------------------- SALVAR --------------------------------------------
# Cria pasta de saída
output_dir = os.path.join(PROCESSED_FOLDER, '{}-urlPrincipal'.format(DATASET_FOLDER))
os.makedirs(output_dir, exist_ok=True)

# Salva bases
for df, file_name in [(df_interactions, FILE_INTERACTIONS), (df_bookmarks, FILE_ITEMS), (df_users, FILE_CONTACTS), (df_tags, FILE_TAGS)]:
    df.to_csv(os.path.join(output_dir, file_name), sep=DELIMITER, encoding=ENCODING, quoting=QUOTING, quotechar=QUOTECHAR, header=True, index=False)
print('OK!')

