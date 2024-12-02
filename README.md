# Weighted sims recommender

Official repository for the paper "_Collaborative filtering through weighted similarities of user and item embeddings_", to be published at the 40th ACM/SIGAPP Symposium On Applied Computing (2025).

## Abstract

In recent years, neural networks and other complex models have dominated recommender systems, often setting new benchmarks for state-of-the-art performance. Yet, despite these advancements, award-winning research has demonstrated that traditional matrix factorization methods can remain competitive, offering simplicity and reduced computational overhead. Hybrid models, which combine matrix factorization with newer techniques, are increasingly employed to harness the strengths of multiple approaches. This paper proposes a novel ensemble method that unifies user-item and item-item recommendations through a weighted similarity framework to deliver top-N recommendations. Our approach is distinctive in its use of shared user and item embeddings for both recommendation strategies, simplifying the architecture and enhancing computational efficiency. Extensive experiments across multiple datasets show that our method achieves competitive performance and is robust in varying scenarios that favor either user-item or item-item recommendations. Additionally, by eliminating the need for embedding-specific fine-tuning, our model allows for the seamless reuse of hyperparameters from the base algorithm without sacrificing performance. This results in a method that is both efficient and easy to implement.

## Key achievements

* Efficient ensemble method that combines user-item and item-item similarity to yield the recommendation;
* Uses the same item and user embeddings for both types of similarity calculation;
* Any vector representation can be employed, e.g., matrix factorization (ALS or BPR) and neural networks (RecVAE);
* Marginal gains when fine-tuning the embeddings, allowing reuse of pre-trained embeddings without significant loss of quality.

Below we present a summary of the method and the achieved results. More details can be found in the original paper.

## Visual representation


![algorithm-diagram](https://github.com/UFSCar-LaSID/weighted-sims-recommender/blob/main/images/algo-diagram.png)

Visual representation of using weighted similarity of embeddings (our proposed algorithm), with items consumed by the user represented in $${\color{green}green}$$, and similar items denoted by the dashed arrows. When using a traditional user-item similarity recommender, only items close to the user are selected (1), which may include items that are similar to the user but unrelated to their consumed items (2). When using the hybrid model with an item-item similarity, items distant from the user but close to their consumption history would also be recommended (3). The final recommendation would then be composed of items similar to both the user and their consumed items, as represented in $${\color{blue}blue}$$.

## Results

In this section we show the results obtained by our algorithm. In the Hit-Rate and NDCG tables our proposal is named as "Weighted". In the NDCG graphic the "WS" lines are our algorithm results. Consult our paper for more details.

### Hit-Rate@10 achieved by each algorithm across all datasets:

![HitRate-table](https://github.com/ReisPires/weighted-sims/blob/main/images/HR-table.png)

### NDCG@10 achieved by each algorithm across all datasets:

![NDCG-table](https://github.com/ReisPires/weighted-sims/blob/main/images/NDCG-table.png)

### NDCG@N with N ranging from 1 to 20. Recommender UI corresponds to user-item, II to item-item, and WS to the weighted similarities ensemble:

![NDCG-graphic](https://github.com/ReisPires/weighted-sims/blob/main/images/ndcg_color_by_algo-no-border.svg)

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

OBS 2: Due to some of the used packages, this will not work on Windows (it will only work with WSL)

### Installing with Docker

To install the libraries with Docker, execute the following steps:

1- Build a Docker image:

```
docker build -t weighted-sims .
```

2- Run the Docker container:

```
docker run -it --gpus all --shm-size=8g \
    -v <path-to-datasets>:/weighted-sims/datasets \
    -v <path-to-raw>:/weighted-sims/raw \
    -v <path-to-results>:/weighted-sims/results \
    weighted-sims /bin/bash -c "source activate py38 && /bin/bash"
```

Replace the `<path-to-datasets>` with an absolute path to save the preprocessed datasets on your machine.

Replace `<path-to-raw>` with an absolute path to raw datasets

Replace `<path-to-results>` with an absolute path to save the results on your machine.

Inside the container, it's possible to execute the scripts from this repository.

## Datasets

Downloading the datasets is necessary to run the experiments. A list with download link and where to save the files are given below:

- [Anime Recommendations](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database?select=rating.csv): put `rating.csv` file in `raw/AnimeRecommendations`
- [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data?select=train.csv): put `train.csv` file in `raw/BestBuy`
- [Ciao DVD](https://guoguibing.github.io/librec/datasets/CiaoDVD.zip): put `movie-ratings.txt` file in `raw/CiaoDVD`
- [Delicious Bookmarks](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip): put `user_taggedbookmarks.dat` file in `raw/DeliciousBookmarks`
- [Film Trust](https://guoguibing.github.io/librec/datasets/filmtrust.zip): put `ratings.txt` file in `raw/FilmTrust`
- [Jester](https://eigentaste.berkeley.edu/dataset/): download [jester_dataset_1_1.zip](https://eigentaste.berkeley.edu/dataset/jester_dataset_1_1.zip), [jester_dataset_1_2.zip](https://eigentaste.berkeley.edu/dataset/jester_dataset_1_2.zip), [jester_dataset_1_3.zip](https://eigentaste.berkeley.edu/dataset/jester_dataset_1_3.zip) and [jester_dataset_2+.zip](https://eigentaste.berkeley.edu/dataset/archive/jester_dataset_3.zip). Put the unziped files in `raw/Jester`
- [Last.FM](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip): put `user_artists.dat` file in `raw/LastFM`
- [MovieLens 1M](https://files.grouplens.org/datasets/movielens/ml-1m.zip): put `ratings.dat` file in `raw/MovieLens-1M`
- [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset): put `events.csv` file in `raw/RetailRocket`

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

Replace `<datasets>` with the names (or indexes) of the datasets separated by space. The available datasets to preprocess are:

- \[1\]: AnimeRecommendations
- \[2\]: BestBuy
- \[3\]: CiaoDVD
- \[4\]: DeliciousBookmarks
- \[5\]: FilmTrust
- \[6\]: Jester
- \[7\]: LastFM
- \[8\]: AnimeRecommendations
- \[9\]: RetailRocket-transactions
- all (it will use all datasets)

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
- \[5\]: FilmTrust
- \[6\]: Jester
- \[7\]: LastFM
- \[8\]: AnimeRecommendations
- \[9\]: RetailRocket-transactions
- all (it will use all datasets)

Replace `<algorithms>` with the names (or indexes) separated by space of the algorithms. The available algorithms are:

- \[1\]: ALS
- \[2\]: BPR
- \[3\]: RecVAE
- \[4\]: ALS_itemSim
- \[5\]: BPR_itemSim
- \[6\]: RecVAE_itemSim
- \[7\]: ALS_weighted
- \[8\]: BPR_weighted
- \[9\]: RecVAE_weighted
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
- \[5\]: FilmTrust
- \[6\]: Jester
- \[7\]: LastFM
- \[8\]: AnimeRecommendations
- \[9\]: RetailRocket-transactions
- all (it will use all datasets)

Replace `<algorithms>` with the names (or indexes) of the algorithms separated by space. The available algorithms are:

- \[1\]: ALS
- \[2\]: BPR
- \[3\]: RecVAE
- \[4\]: ALS_itemSim
- \[5\]: BPR_itemSim
- \[6\]: RecVAE_itemSim
- \[7\]: ALS_weighted
- \[8\]: BPR_weighted
- \[9\]: RecVAE_weighted
- all (it will use all algorithms)

### Generating graphics

To generate the same graphics from our paper, execute all cells in `src/graphics.ipynb` Jupyter Notebook.

## Citation

If our weighted sims recommender is useful or relevant to your research, please kindly recognize our contributions by citing our paper (_to be updated after publication_):

```bib
@inproceedings{pires2025,
  title={Collaborative filtering through weighted similarities of user and item embeddings},
  author={Pedro Pires and Rafael Tofoli and Gregorio Fornetti and Tiago Almeida},
  booktitle={Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
  series={SAC 2025},
  year={2025}
}
```
