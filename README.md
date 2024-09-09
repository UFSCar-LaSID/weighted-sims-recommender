# weighted-sims

Official repository for paper "Collaborative filtering through weighted similarities of matrix factorization embeddings"

## Datasets

To run the experiments, it's necessary to download the datasets. A list with download link and where to save the files are given bellow:

- [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset): put the downloaded files in `raw/retailrocket`

## Installing

The supported plataforms for executing the code are the following:

- macOS 10.12+ x86_64.
- Linux x86_64 (including WSL on Windows 10).

There are two ways to install the libs: (1) installing manually and (2) using Docker (recommended, and works for Windows too).

### Installing manually

Executing the command above will install the necessary libraries:

```
pip install -r requirements.txt
```

OBS 1: It's recommended to use a new conda environment before doing it. That way you can prevent from breaking library versions for other codes of yours.

OBS 2: This will not work on Windows (it will only work with WSL)

### Installing with Docker

To install the libraries with Docker, execute the following steps:

1- Build a Docker image:

```
docker build -t weighted-sims .
```

2- Run the Docker container:

```
docker run -it \
    -v <path-to-datasets>:/weighted-sims/datasets \
    -v <path-to-raw>:/weighted-sims/raw \
    -v <path-to-results>:/weighted-sims/results \
    weighted-sims /bin/bash
```

Replace the `<path-to-datasets>` with a absolute path to save the pre-processed datasets on your machine, replace `<path-to-raw>` with a absolute path to raw datasets and replace `<path-to-results>` with a absolute path to save the results on your machine.

Inside the container it's possible to execute the scripts from this repository.

## Executing the code

Execute the following scripts to reproduce our results:

### Dataset preprocess

With the raw datasets downloaded (more details in Dataset LINK AQUIII), it's necessary to preprocess them before generating the recommendations.
To do that, execute the following command:

```
python src/scripts/preprocess.py
```

Executing this python code, it will ask you which datasets to preprocess. Input the datasets indexes separated by space to select the datasets.

Another way to select the datasets is executing the command bellow:

```
python src/scripts/preprocess.py --datasets <datasets>
```

Replace `<datasets>` with the names (or indexes) separated by space of the datasets. The available datasets to preprocess are:

- \[1\]: RetailRocket
- all (it will preprocess all datasets available)

### Train and generate recommendations

To train and generate the recommendations for the test folds, execute the following command:

`python src/scripts/generate_recommendations.py`

When executed, it will prompt you to input the recommenders that will be trained in which datasets. It's possible to select more than one recommender and dataset per execution, just type the indexes of datasets and recommenders separated by space. The selected recommenders will be trained in all selected datasets.

Another way to supply the inputs is by command line. To do that, execute the command above:

`python src/scripts/generate_recommendations.py --datasets <datasets> --algorithms <algorithms>`

Replace `<datasets>` with the names (or indexes) separated by space of the datasets. The available datasets are:

- \[1\]: AnimeRecommendations
- \[2\]: BestBuy
- \[3\]: CiaoDVD
- \[4\]: DeliciousBookmarks
- \[5\]: Filmtrust
- \[6\]: Jester
- \[7\]: Last.FM-Listened
- \[8\]: AnimeRecommendations
- \[9\]: RetailRocket-Transactions
- all (it will use all datasets)

Replace `<algorithms>` with the names (or indexes) separated by space of the algorithms. The available algorithms are:

- \[1\]: ALS
- \[2\]: BPR
- \[3\]: ALS_itemSim
- \[4\]: BPR_itemSim
- \[5\]: ALS_weighted
- \[6\]: BPR_weighted
- all (it will use all algorithms)

### Evaluate: calculate metrics

To calculate metrics for the executed algoritms from the previous code, execute the following command:

`python src/scripts/evaluate.py`

When executed, it will prompt you to input the recommenders that will be evaluated in which datasets. It's possible to select more than one recommender and dataset per execution, just type the indexes of datasets and recommenders separated by space. The selected recommenders will be evaluated in all selected datasets.

Another way to supply the inputs is by command line. To do that, execute the command above:

`python src/scripts/evaluate.py --datasets <datasets> --algorithms <algorithms>`

Replace `<datasets>` with the names (or indexes) separated by space of the datasets. The available datasets are:

- \[1\]: AnimeRecommendations
- \[2\]: BestBuy
- \[3\]: CiaoDVD
- \[4\]: DeliciousBookmarks
- \[5\]: Filmtrust
- \[6\]: Jester
- \[7\]: Last.FM-Listened
- \[8\]: AnimeRecommendations
- \[9\]: RetailRocket-Transactions
- all (it will use all datasets)

Replace `<algorithms>` with the names (or indexes) separated by space of the algorithms. The available algorithms are:

- \[1\]: ALS
- \[2\]: BPR
- \[3\]: ALS_itemSim
- \[4\]: BPR_itemSim
- \[5\]: ALS_weighted
- \[6\]: BPR_weighted
- all (it will use all algorithms)

### Generating graphics

To generate the same graphics from our paper, execute all cells in `src/graphics.ipynb` Jupyter Notebook.