# weighted-sims

Official repository for paper "Collaborative filtering through weighted similarities of matrix factorization embeddings"

## Datasets

To run the experiments, it's necessary to download the datasets. A list with download link and where to save the files are given bellow:

- [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset): put the downloaded files in `datasets/RetailRocket`

## Installing

The supported plataforms for executing the code are the following:

- macOS 10.12+ x86_64.
- Linux x86_64 (including WSL on Windows 10).

There are two ways to install the libs: (1) installing manually and (2) using Docker (recommended, and works for Windows too).

### Installing manually

Executing the command above will install the necessary libraries:

`pip install -r requirements.txt`

OBS 1: It's recommended to use a new conda environment before doing it. That way you can prevent from breaking library versions for other codes of yours.

OBS 2: This will not work on Windows (it will only work with WSL)

### Installing with Docker

To install the libraries with Docker, execute the following steps:

1- Build a Docker image:

`docker build -t weighted-sims .`

2- Run the Docker container:

`docker run -it weighted-sims /bin/bash`

Inside the container it's possible to execute the scripts from this repository.

## Executing the code

