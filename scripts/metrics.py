import os.path
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

class Metrics:
    def __init__(self, folds, n_eval):
        self.folds = folds
        self.n_eval = n_eval
        column_names = ['Parameters'] + ['{}@{}'.format(m, n+1) for m in ['Prec', 'Rec', 'F1_Score', 'Hit_Rate', 'NDCG'] for n in range(self.n_eval)]
        self.result_df = pd.DataFrame([], columns=column_names)


    def get_dataframe(self):
        return self.result_df


    def precision_recall_hitrate(self, real, predicted, k=None):
        prec = rec = hr = 0
        n_samples = len(real)
        for i in range(n_samples):
            intersect = len(set(real[i]).intersection(predicted[i,:k]))
            prec += intersect / k
            rec += intersect / len(real[i])
            hr += bool(intersect)
        prec /= n_samples
        rec /= n_samples
        hr /= n_samples
        return prec, rec, hr


    def f1_score(self, prec, rec):       
        return 2 * np.divide((prec*rec), (prec+rec), out=np.zeros_like(prec), where=(prec*rec)>0)
    
    
    def ndcg_score(self, actual, predicted, k):
        dcg, idcg = 0, 0
        for real, pred in zip(actual, predicted):
            relevance = {item: i for i, item in enumerate(real)}
            dcg += sum(1 / np.log2(i + 2) for i, item in enumerate(pred[:k]) if relevance.get(item) is not None)
            idcg += sum(1 / np.log2(i + 2) for i in range(len(real)))
        return (dcg / idcg)


    def add_metrics(self, recomendation_filepath):

        for parameters in tqdm(os.listdir(recomendation_filepath)):            

            for fold in range(1, self.folds+1):
                
                data = pd.read_csv(
                    os.path.join(recomendation_filepath, parameters, str(fold), 'recommendations.csv'), 
                    sep=';', 
                    converters={"recommendations": ast.literal_eval, "items": ast.literal_eval}
                )

                previstos_array = np.array(data['recommendations'].tolist())
                reais_array = data['items'].tolist()

                prec = np.zeros(self.n_eval)
                rec = np.zeros(self.n_eval)                
                hr = np.zeros(self.n_eval)
                ndcg = np.zeros(self.n_eval)
                
                for n in range(self.n_eval):
                    fold_prec, fold_rec, fold_hr = self.precision_recall_hitrate(reais_array, previstos_array, n+1)
                    prec[n] += fold_prec
                    rec[n] += fold_rec
                    hr[n] += fold_hr
                    ndcg[n] += self.ndcg_score(reais_array, previstos_array, n+1)

            prec /= self.folds
            rec /= self.folds
            hr /= self.folds
            ndcg /= self.folds
            
            f1 = self.f1_score(prec, rec)

            metrics = np.concatenate([prec, rec, f1, hr, ndcg])

            self.result_df.loc[len(self.result_df)] = [parameters] + metrics.tolist()


    def save_metrics(self, dataset_name, recommender_name):
        filedir = os.path.join('results', 'metrics', dataset_name, recommender_name)
        os.makedirs(filedir, exist_ok=True)
        filepath = os.path.join(filedir, 'metrics.csv')
        self.result_df.to_csv(filepath, sep=';', index=False)
