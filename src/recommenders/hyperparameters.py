import src as kw

ALS_HYPERPARAMETERS = {
    'factors': [32, 64, 128],
    'regularization': [0.001, 0.01, 0.1],
    'iterations': [15, 30, 50]
}

BPR_HYPERPARAMETERS = {
    'factors': [128],
    'learning_rate': [0.01],
    'regularization': [0.01],
    'iterations': [50]
}

REC_VAE_HYPERPARAMETERS = {
    'gamma': [0.0035, 0.005, 0.01]
}

ALS_ITEM_SIM_HYPERPARAMETERS = {

}

BPR_ITEM_SIM_HYPERPARAMETERS = {

}

REC_VAE_ITEM_SIM_HYPERPARAMETERS = {

}

ALS_WEIGHTED_SIM_HYPERPARAMETERS = {
    'similarity_weights': [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2)],
    'similarity_metric': ['cosine', 'dot']
}

BPR_WEIGHTED_SIM_HYPERPARAMETERS = {
    'similarity_weights': [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2)],
    'similarity_metric': ['cosine', 'dot']
}

REC_VAE_WEIGHTED_HYPERPARAMETERS = {
    'similarity_weights': [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2)],
    'similarity_metric': ['cosine', 'dot']
}

ALS_MEAN_SIM_HYPERPARAMETERS = {

}

BPR_MEAN_SIM_HYPERPARAMETERS = {

}