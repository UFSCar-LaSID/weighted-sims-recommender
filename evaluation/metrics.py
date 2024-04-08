import os.path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from rankereval import NumericLabels, Rankings, NDCG
import ast

class Metrics:
    def __init__(self, folds, k_array):
        self.folds = folds
        self.k_array = k_array
        self.result_df = pd.DataFrame()

    def get_dataframe(self):
        return self.result_df

    def precision(self, real, predicted, k=None):
        precision_aux = 0
        for i in range(len(real)):
            precision_aux += len(set(real[i]).intersection(predicted[i,:k])) / len(predicted[i,:k])
        return precision_aux / len(real)

    def recall(self, real, predicted, k=None):
        recall_aux = 0
        for i in range(len(real)):
            recall_aux += len(set(real[i]).intersection(predicted[i,:k])) / len(real[i])
        return recall_aux / len(real)

    def f1_score(self, prec, rec):
        if prec + rec == 0:
            return 0
        return 2 * (prec * rec) / (prec + rec)

    def hit_rate(self, real, predicted, k=None):
        hits = 0
        for i in range(len(real)):
            if set(real[i]).intersection(predicted[i,:k]):
                hits += 1
        return hits / len(real)
    
    import numpy as np
    
    def ndcg(self, actual, predicted, k):
    
        dcg, idcg = 0, 0
        
        for real, pred in zip(actual, predicted):
            relevance = {item: i for i, item in enumerate(real)}
            dcg += sum(1 / np.log2(i + 2) for i, item in enumerate(pred[:k]) if relevance.get(item) is not None)
            idcg += sum(1 / np.log2(i + 2) for i in range(len(real))
                    
        return (dcg / idcg)

    #Gera metricas a partir do arquivo de recomendações a partir do filepath dado, concatena o resultado no dataframe final
    def add_metrics(self, embeddings_filepath):

        result_aux = pd.DataFrame()

        for k in self.k_array:

            prec, rec, f1_score_var, hit_rate_var = 0, 0, 0, 0

            parameters = embeddings_filepath.split("/")[4]
            
            for fold in range(1, self.folds+1):

                data = pd.read_csv(open(os.path.join(os.path.dirname(__file__), '..', 
                                embeddings_filepath, str(fold), 
                                'recommendations.csv'), 'rb'), sep=';',
                                converters={"recommendations": ast.literal_eval, "items": ast.literal_eval})

                # Converte dataframe em arrays numpy
                previstos_array = np.array(data['recommendations'].tolist())
                reais_array = np.array(data['items'].tolist())

                #Realiza o calculo das metricas
                prec += self.precision(reais_array, previstos_array, k)
                rec += self.recall(reais_array, previstos_array, k)
                f1_score_var += self.f1_score(prec, rec)
                hit_rate_var += self.hit_rate(reais_array, previstos_array, k)
                ndcg_var = self.ndcg(reais_array, previstos_array, k)

            #Divide pelo numero de Folds e gera o nome das colunas
            metrics = [[prec/self.folds, rec/self.folds, f1_score_var/self.folds, hit_rate_var/self.folds, ndcg_var/self.folds]]
            names  = ['Prec@' + str(k), 'Rec@' + str(k), 'F1_Score@' + str(k), 'Hit_Rate@' + str(k), 'NDCG@' + str(k)]


            df_aux = pd.DataFrame(data=metrics, columns=names)
            result_aux = pd.concat([result_aux, df_aux], axis=1)

        result_aux.insert(0, 'Parameters', parameters)
        self.result_df = pd.concat([self.result_df, result_aux], axis=0)

    def save_metrics(self, dataset_name, recommender_name):

        filedir = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', dataset_name, recommender_name)
        os.makedirs(filedir, exist_ok=True)
        filepath = os.path.join(filedir, 'metrics.csv')
        self.result_df.to_csv(filepath, sep=';', index=False)