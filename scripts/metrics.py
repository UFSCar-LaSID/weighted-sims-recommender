import os.path
import pandas as pd
import numpy as np
import ast

class Metrics:
    def __init__(self, folds, k_array):
        self.folds = folds
        self.k_array = k_array
        self.result_df = pd.DataFrame()


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
        if prec + rec == 0:
            return 0
        return 2 * (prec * rec) / (prec + rec)
    
    
    def ndcg(self, actual, predicted, k):
        dcg, idcg = 0, 0
        for real, pred in zip(actual, predicted):
            relevance = {item: i for i, item in enumerate(real)}
            dcg += sum(1 / np.log2(i + 2) for i, item in enumerate(pred[:k]) if relevance.get(item) is not None)
            idcg += sum(1 / np.log2(i + 2) for i in range(len(real)))
        return (dcg / idcg)


    #Gera metricas a partir do arquivo de recomendações a partir do filepath dado, concatena o resultado no dataframe final
    def add_metrics(self, recomendation_filepath):

        result_aux = pd.DataFrame()

        for parameters in os.listdir(recomendation_filepath):

            for k in self.k_array:

                prec = rec = f1_score_var = hr = ndcg_var = 0.0            
            
                for fold in range(1, self.folds+1):

                    data = pd.read_csv(
                        os.path.join(recomendation_filepath, parameters, str(fold), 'recommendations.csv'), 
                        sep=';', 
                        converters={"recommendations": ast.literal_eval, "items": ast.literal_eval}
                    )

                    # Converte dataframe em arrays numpy
                    previstos_array = np.array(data['recommendations'].tolist())
                    reais_array = data['items'].tolist()

                    #Realiza o calculo das metricas
                    fold_prec, fold_rec, fold_hr = self.precision_recall_hitrate(reais_array, previstos_array, k)
                    prec += fold_prec
                    rec += fold_rec
                    hr += fold_hr
                    f1_score_var += self.f1_score(prec, rec)
                    ndcg_var += self.ndcg(reais_array, previstos_array, k)

                #Divide pelo numero de Folds e gera o nome das colunas
                metrics = [[prec/self.folds, rec/self.folds, f1_score_var/self.folds, hr/self.folds, ndcg_var/self.folds]]
                names  = ['Prec@' + str(k), 'Rec@' + str(k), 'F1_Score@' + str(k), 'Hit_Rate@' + str(k), 'NDCG@' + str(k)]

                df_aux = pd.DataFrame(data=metrics, columns=names)
                result_aux = pd.concat([result_aux, df_aux], axis=1)

            result_aux.insert(0, 'Parameters', parameters)
            self.result_df = pd.concat([self.result_df, result_aux], axis=0)


    def save_metrics(self, dataset_name, recommender_name):
        filedir = os.path.join('results', 'metrics', dataset_name, recommender_name)
        os.makedirs(filedir, exist_ok=True)
        filepath = os.path.join(filedir, 'metrics.csv')
        self.result_df.to_csv(filepath, sep=';', index=False)