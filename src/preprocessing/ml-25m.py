# Link para download da base original: https://grouplens.org/datasets/movielens/
import csv
from datetime import datetime
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Pasta de dataset
DATASET_FOLDER = 'ml-25m' # Renomear a pasta para esse nome após o download
RAW_FOLDER = 'raw' # Salvar os dados dentro dessa pasta após o download
PROCESSED_FOLDER = 'processed'

# Nomes de arquivos
FILE_ITEMS = 'items.csv'
FILE_INTERACTIONS = 'interactions.csv'
FILE_GENRES = 'genres.csv'
FILE_ITEM_GENRES = 'item_genres.csv'

# Dados dos CSVs
DELIMITER = ';'
ENCODING = "utf-8"
QUOTING = csv.QUOTE_ALL
QUOTECHAR = '"'

# Nomes das colunas
COLUMN_ITEM_ID = 'id_item'
COLUMN_USER_ID = 'id_user'
COLUMN_RATING = 'rating'
COLUMN_GENRES_ID = 'id_genres'
COLUMN_GENRES_NAMES = 'name_genres'
COLUMN_DATETIME = 'datetime'
COLUMN_TIMESTAMP = 'timestamp'

# Gêneros válidos
GENRES = {
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
}

# Define pasta onde vai salvar
intput_dir = os.path.join(RAW_FOLDER, DATASET_FOLDER)
output_dir = os.path.join(PROCESSED_FOLDER, DATASET_FOLDER)
os.makedirs(output_dir, exist_ok=True)

# ========================================= FILMES =========================================
# Lê CSV de filmes
df_movies = pd.read_csv(os.path.join(intput_dir, 'movies.csv'), sep=',', header=0, index_col=False)

# Separa o ano do título
df_movies['year'] = df_movies['title'].apply(lambda x: x.strip()[-5:-1])
df_movies['title'] = df_movies['title'].apply(lambda x: x.strip()[:-7])

# Gera arquivo de filmes com gêneros
df_movies_genres = df_movies[['movieId', 'genres']].copy()
df_movies_genres['genres'] = df_movies_genres['genres'].str.split('|')
df_movies_genres = df_movies_genres.explode('genres')
df_movies_genres = df_movies_genres[df_movies_genres['genres'].isin(GENRES)]

# Gera arquivo de gêneros
df_genres = df_movies_genres[['genres']]
le_genres = LabelEncoder()
df_genres[COLUMN_GENRES_ID] = le_genres.fit_transform(df_movies_genres['genres'])
df_genres = df_genres.drop_duplicates(keep='first')[[COLUMN_GENRES_ID, 'genres']].sort_values(COLUMN_GENRES_ID)
df_genres.columns = [COLUMN_GENRES_ID, COLUMN_GENRES_NAMES]

# Arruma arquivo de filmes com gêneros
df_movies_genres['genres'] = le_genres.transform(df_movies_genres['genres'])
df_movies_genres.columns = [COLUMN_ITEM_ID, COLUMN_GENRES_ID]
df_movies_genres = df_movies_genres.sort_values([COLUMN_ITEM_ID, COLUMN_GENRES_ID])

# Remove coluna de gêneros dos filmes e atualiza índice
df_movies = df_movies.drop(columns=['genres']).rename(columns={'movieId': COLUMN_ITEM_ID})

# ======================================= INTERACOES =======================================
# Lê CSV de interações
df_interactions = pd.read_csv(os.path.join(intput_dir, 'ratings.csv'))
df_interactions.columns = [COLUMN_USER_ID, COLUMN_ITEM_ID, COLUMN_RATING, COLUMN_TIMESTAMP]
# Gera datetimes
df_interactions[COLUMN_DATETIME] = df_interactions[COLUMN_TIMESTAMP].apply(lambda x: str(datetime.fromtimestamp(x)))

# ========================================= SALVAR =========================================
# Salva bases
for df, file_name in [(df_movies, FILE_ITEMS), (df_movies_genres, FILE_ITEM_GENRES), (df_genres, FILE_GENRES), (df_interactions, FILE_INTERACTIONS)]:
    df.to_csv(os.path.join(output_dir, file_name), sep=DELIMITER, encoding=ENCODING, quoting=QUOTING, quotechar=QUOTECHAR, header=True, index=False)
print('OK!')