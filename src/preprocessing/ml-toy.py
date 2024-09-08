import pandas as pd
import os

df_interactions = pd.read_csv('1-datasets/ml-25m/interactions.csv', sep=';')
df_items = pd.read_csv('1-datasets/ml-25m/items.csv', sep=';')

most_consumed_items_df = df_interactions.groupby('id_item').size().sort_values(ascending=False)
most_consumed_items_df = most_consumed_items_df.reset_index()

top_30_items = most_consumed_items_df.head(30)

top_30_items_df = df_items[df_items['id_item'].isin(top_30_items['id_item'])]

interactons_with_top_30_items = df_interactions[df_interactions['id_item'].isin(top_30_items['id_item'])]

os.makedirs('1-datasets/ml-toy', exist_ok=True)
interactons_with_top_30_items.to_csv('1-datasets/ml-toy/interactions.csv', sep=';', index=False)
top_30_items_df.to_csv('1-datasets/ml-toy/items.csv', sep=';', index=False)