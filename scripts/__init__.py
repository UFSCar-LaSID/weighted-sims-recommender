import csv

# Nomes de arquivos da base de dados
FILE_ITEMS = 'items.csv'
FILE_USERS = 'users.csv'
FILE_INTERACTIONS = 'interactions.csv'

# Colunas da tabela de dataset
DATASET_ID = 'id'
DATASET_NAME = 'name'
DATASET_TYPE = 'type'
DATASET_GDRIVE = 'gdrive'

# Dados dos CSVs
DELIMITER = ';'
QUOTECHAR = '"'
QUOTING = csv.QUOTE_ALL
ENCODING = "ISO-8859-1"

# Colunas dos CSVs e dataframmes
COLUMN_ITEM_ID = 'id_item'
COLUMN_ITEM_NAME = 'name_item'
COLUMN_USER_ID = 'id_user'
COLUMN_USER_NAME = 'name_user'
COLUMN_RATING = 'interaction'

# Colunas do arquivo de log
LOG_COLUMN_USER = 'user'
LOG_COLUMN_ITEMS = 'items'
LOG_COLUMN_RECOMMENDATIONS = 'recommendations'

# Infos da recomendação
TOP_N = 10