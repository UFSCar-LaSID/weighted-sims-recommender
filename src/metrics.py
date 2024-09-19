import os.path
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import itertools
from scipy.sparse import csr_matrix
import src as kw



class Metrics:
    def __init__(self, folds, n_eval):
        self.folds = folds
        self.n_eval = n_eval
        column_names = ['Parameters'] + ['{}@{}'.format(m, n+1) for m in ['Prec', 'Rec', 'F1_Score', 'Hit_Rate', 'NDCG'] for n in range(self.n_eval)]
        self.result_df = pd.DataFrame([], columns=column_names)


    def get_dataframe(self):
        return self.result_df
    
    def precision_recall_hitrate(self, consumed_matrix, recs_matrix, consumed_sizes, recs_encoded, k=None):
        recs_matrix_top_k = recs_matrix.copy()
        recs_matrix_top_k[np.repeat(np.arange(len(consumed_sizes)), kw.TOP_N - k), recs_encoded[:, k:].reshape(-1)] = 0
        
        intersect_sum = np.array(consumed_matrix.multiply(recs_matrix_top_k).sum(axis=1)).reshape(-1)

        prec = (intersect_sum / k).mean()
        rec = (intersect_sum / consumed_sizes).mean()
        hr = (intersect_sum >= 1).mean()
        
        return prec, rec, hr


    # Funciona para um vetor de precisoes e revocacoes
    def f1_score(self, prec, rec):       
        return 2 * np.divide((prec*rec), (prec+rec), out=np.zeros_like(prec), where=(prec*rec)>0)

    def transform_to_sparse_matrix(self, recs, consumed, k):
        flat_consumed = np.array(list(itertools.chain(*consumed)))
        flat_recomended = recs.flatten()

        encoder = LabelEncoder()
        encoder.fit(np.concatenate([flat_consumed, flat_recomended]))

        sizes = np.array([len(consumed_list) for consumed_list in consumed])
        consumed_flatten_encoded = encoder.transform(flat_consumed)
        indexes_consumed = np.repeat(np.arange(len(consumed)), repeats=sizes)
        consumed_matrix = csr_matrix((np.ones(len(indexes_consumed)), (indexes_consumed, consumed_flatten_encoded)), shape=(len(consumed), encoder.classes_.shape[0]), dtype=bool)

        
        indexes_recs = np.repeat(np.arange(len(recs)), repeats=k)
        recs_flatten_encoded = encoder.transform(flat_recomended)
        recs_matrix = csr_matrix((np.ones(len(indexes_recs)), (indexes_recs, recs_flatten_encoded)), shape=(len(consumed), encoder.classes_.shape[0]), dtype=bool)

        recs_encoded = recs_flatten_encoded.reshape(-1, k)

        return consumed_matrix, recs_matrix, sizes, recs_encoded
    
    def create_vectors(self, sizes) -> np.ndarray:
        # Repeat each element in sizes by its value and create a cumulative range
        return np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)

    
    def ndcg_score(self, consumed_matrix, consumed_sizes, recs_encoded, k=None):
        possible_scores = np.array([1 / np.log2(i+2) for i in range(k)])

        idcg = possible_scores[self.create_vectors(np.clip(consumed_sizes, a_max=k, a_min=1))].sum()
        dcg = possible_scores[np.where(consumed_matrix[np.repeat(np.arange(len(consumed_sizes)), k), recs_encoded[:, :k].reshape(-1)].reshape(-1, k) == True)[1]].sum()

        return dcg / idcg

    #Gera metricas a partir do arquivo de recomendações a partir do filepath dado, concatena o resultado no dataframe final
    def add_metrics(self, recomendation_filepath):

        for parameters in tqdm(os.listdir(recomendation_filepath)):

            prec = np.zeros(self.n_eval)
            rec = np.zeros(self.n_eval)                
            hr = np.zeros(self.n_eval)
            ndcg = np.zeros(self.n_eval)          

            for fold in range(1, self.folds+1):
                
                data = pd.read_csv(
                    os.path.join(recomendation_filepath, parameters, str(fold), 'recommendations.csv'), 
                    sep=';', 
                    converters={"recommendations": ast.literal_eval, "items": ast.literal_eval}
                )

                # Converte dataframe em arrays numpy
                previstos_array = np.array(data['recommendations'].tolist())
                reais_array = data['items'].to_list()

                consumed_matrix, recs_matrix, consumed_sizes, recs_encoded = self.transform_to_sparse_matrix(
                    previstos_array, reais_array, kw.TOP_N
                )
                
                # Realiza o calculo das metricas para top-N
                for n in range(self.n_eval):
                    fold_prec, fold_rec, fold_hr = self.precision_recall_hitrate(consumed_matrix, recs_matrix, consumed_sizes, recs_encoded, n+1)
                    prec[n] += fold_prec
                    rec[n] += fold_rec
                    hr[n] += fold_hr
                    ndcg[n] += self.ndcg_score(consumed_matrix, consumed_sizes, recs_encoded, n+1)

            # Divide metricas pelo numero de folds 
            prec /= self.folds
            rec /= self.folds
            hr /= self.folds
            ndcg /= self.folds
            
            # Calcula F-Medida
            f1 = self.f1_score(prec, rec)

            # Agrupa metricas
            metrics = np.concatenate([prec, rec, f1, hr, ndcg])

            # Insere a linha
            self.result_df.loc[len(self.result_df)] = [parameters] + metrics.tolist()


    def save_metrics(self, dataset_name, recommender_name):
        filedir = os.path.join('results', 'metrics', dataset_name, recommender_name)
        os.makedirs(filedir, exist_ok=True)
        filepath = os.path.join(filedir, 'metrics.csv')
        self.result_df.to_csv(filepath, sep=';', index=False)
        return filepath

    def print_best_results(self):
        best_column = 'NDCG@10'
        best_row = self.result_df.loc[self.result_df[best_column].idxmax()]
        print(f'Os melhores hiperparâmetros foram: {best_row["Parameters"]}')
        results_list = []
        for i in range(self.n_eval):
            results_list.append([
                i+1,
                best_row['Prec@{}'.format(i+1)],
                best_row['Rec@{}'.format(i+1)],
                best_row['F1_Score@{}'.format(i+1)],
                best_row['Hit_Rate@{}'.format(i+1)],
                best_row['NDCG@{}'.format(i+1)]
            ])
        final_table = pd.DataFrame(results_list, columns=['Top-N', 'Prec', 'Rec', 'F1_Score', 'Hit_Rate', 'NDCG'])

        print(final_table[:self.n_eval])
