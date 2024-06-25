# Collaborative filtering through weighted similarities of matrix factorization embeddings

Code used for the experiments reported in the article *Collaborative filtering through weighted similarities of matrix factorization embeddings*, submitted for the **Brazilian Conference on Intelligent Systems (BRACIS 2024)**.

---

Description of files:
* _**requirements.txt**_: used libraries and their corresponding versions;
* _**requirements_all.txt**_: used libraries and all its dependencies;
* _**main.py**_: code for the experiment, with every auxiliary code, as well as the implementation of the algorithms, inside folder *scripts*.

---

Instructions:
* Datasets must be downloaded, preprocessed, and stored in a folder named *datasets*;
* Experiments can be run with ```python3 main.py```. The execution of specific datasets and models can be specified in lists *DATASETS* and *RECOMMENDERS*, respectively;
* Proposed model can be found in ```scripts/recommenders/weightedSim.py```;
* Experiment code will generate three folders inside the folder *results*:
  * ```embeddings```: containing the matrix factorization embeddings in numpy nd.array;
  * ```recommendations```: CSVs representing the recommended items for every dataset, hyperparameter combination, and fold;
  * ```metrics```: CSVs containing the Precision, Recall, F1-Score, Hit Rate and NDCG for the top-N recommendation with N ranging from 1 to 20.
