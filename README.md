# Unified Embeddings PyTorch Implementation

This project is a simple implementation of a Unified Embedding paper. 

The model uses a single embedding table to encode multiple different categorical features in order to save memory and computation time when dealing with large-scale systems.

You can find a [detailed explanation of the code here](). 

## Installing / Getting started

To get started with this project, you need to have Python and PyTorch installed. You can install the required packages using pip:

```shell
pip install torch polars xxhash
```

The above command installs PyTorch, Polars, and xxhash. 

PyTorch is used for creating and training the neural network model, Polars for data manipulation, and xxhash for hashing.

## Features

The main features of this project include:
* Unified embedding layer (ue.py)
* Simple feed-forward neural network for prediction (test.py)
* Training and validation loop for testing the code (test.py)

## Links

- [Unified Embeddings paper](https://arxiv.org/abs/2305.12102)
- [HashedNets](https://github.com/jfainberg/hashed_nets)
- [xxhash](https://pypi.org/project/xxhash/)

## Licensing

The code in this project is licensed under MIT license.