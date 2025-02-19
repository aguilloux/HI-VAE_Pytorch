# HI-VAE

This repository contains the implementation of our Heterogeneous Incomplete Variational Autoendoder model (HI-VAE). It has been written in Python, using PyTorch.

The details of this model are included in this [paper](https://arxiv.org/abs/1807.03653). Please cite it if you use this code for your own research.

## Databse description

There are three different datasets considered in the experiments (Wine, Adult and Default Credit). Each dataset has each own folder, containing:

* **data.csv**: the dataset
* **data_types.csv**: a csv containing the types of that particular dataset. Every line is a different attribute containing three paramenters:
	* type: real, pos (positive), cat (categorical), ord (ordinal), count
	* dim: dimension of the variable
	* nclass: number of categories (for cat and ord)
* **Missingxx_y.csv**: a csv containing the positions of the different missing values in the data. Each "y" mask was generated randomly, containing a "xx" % of missing values.

You can add your own datasets as long as they follow this structure.

## Files description

* **script_HIVAE.sh**: A script with a simple example on how to run the models.
* **src.py**: Contains the HI-VAE models (factorized encoder or input dropout encoder).
* **utils.py**: In this file contains different scripts to support load data, compute likelihood, compute error.
* **experiments.py**: Contains the experiments for the HIVAE models.
* **tutorial.ipynb**: Tutorial notebook.


# Code Pre-requisites

First,
```console
$ pip install virtualenv
$ cd HI-VAE
$ virtualenv -p python3 _venv
$ source _venv/bin/activate
$ pip install -r pip_requirements.txt
$ chmod +x script_HIVAE.sh
```

Then, run
```console
$ ./script_HIVAE.sh
```
