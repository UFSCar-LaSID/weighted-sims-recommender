import scripts as kw

ALS_HYPERPARAMETERS = {
    'factors': [32, 64, 128],
    'regularization': [0.001, 0.01, 0.1],
    'iterations': [15, 30, 50]
}

BPR_HYPERPARAMETERS = {
    'factors': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'regularization': [0.001, 0.01, 0.1],
    'iterations': [50, 100, 200]
}