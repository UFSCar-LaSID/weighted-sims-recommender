# weighted-sims

Official repository for paper "Collaborative filtering through weighted similarities of matrix factorization embeddings"

## Datasets

Downloading the datasets is necessary to run the experiments. A list with download link and where to save the files are given below:

- [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset): put `events.csv` file in `raw/RetailRocket`
- [Film Trust](https://guoguibing.github.io/librec/datasets/filmtrust.zip): put `ratings.txt` file in `raw/FilmTrust`
- [Ciao DVD](https://guoguibing.github.io/librec/datasets/CiaoDVD.zip): put the downloaded files in `raw/CiaoDVD`
- [MovieLens 1M](https://files.grouplens.org/datasets/movielens/ml-1m.zip): put `ratings.dat` file in `raw/MovieLens-1M`
- [Delicious Bookmarks](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip): put `user_taggedbookmarks.dat` file in `raw/DeliciousBookmarks`
- [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data?select=train.csv): put `train.csv` file in `raw/BestBuy`

## Installing

The supported platforms for executing the code are the following:

- macOS 10.12+ x86_64.
- Linux x86_64 (including WSL on Windows 10).

There are two ways to install the libs: (1) installing manually and (2) using Docker (it works for Windows too).

### Installing manually

Executing the command below will install the necessary libraries:

```
pip install -r requirements.txt
```

OBS 1: It's recommended to use a new conda environment before doing it to prevent breaking library versions of other codes.

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

Replace the `<path-to-datasets>` with an absolute path to save the preprocessed datasets on your machine.

Replace `<path-to-raw>` with an absolute path to raw datasets

Replace `<path-to-results>` with an absolute path to save the results on your machine.

Inside the container, it's possible to execute the scripts from this repository.

## Executing the code

Execute the following scripts to reproduce our results:

### Dataset preprocess

With the raw datasets downloaded (more details in [Datasets section](#Datasets)), it's necessary to preprocess them before generating the recommendations.
To do that, execute the following command:

```
python src/scripts/preprocess.py
```

Executing this Python code will ask you which datasets to preprocess. Input the datasets indexes separated by space to select the datasets.

Another way to select the datasets is by executing the command below:

```
python src/scripts/preprocess.py --datasets <datasets>
```

Replace `<datasets>` with the names (or indexes) separated by space of the datasets. The available datasets to preprocess are:

- \[1\]: RetailRocket
- all (it will preprocess all datasets available)

### Train and generate recommendations

To train and generate the recommendations for the test folds, execute the following command:

`python src/scripts/generate_recommendations.py`

When executed, it will prompt you to input the recommenders that will be trained in which datasets. Type the indexes of datasets or algorithms separated by space to select more than one dataset or algorithm. The selected recommenders will be trained in all selected datasets.

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

To calculate metrics for the executed algorithms from the previous code, execute the following command:

`python src/scripts/evaluate.py`

When executed, it will prompt you to input the recommenders that will be evaluated in which datasets. Type the indexes of datasets or algorithms separated by space to select more than one dataset or algorithm. The selected recommenders will be evaluated in all selected datasets.

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
