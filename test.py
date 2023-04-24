
import numpy as np
import pandas as pd

df = pd.read_csv('datasets/RetailRocket-Transactions/interactions_implicit.csv', sep=';')[['id_user', 'id_item']].sample(3000)

from scripts.recommender import ALS

rec = ALS('embeddings', 43)
rec.fit(df)
items_recommended, score = rec.recommend(df)

userid = df['id_user'].unique()

recommendations = pd.DataFrame(np.vstack([np.repeat(userid, 10), np.ravel(items_recommended)]).T, columns=['id_user', 'id_item'])
print(recommendations)