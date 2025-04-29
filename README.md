# HI-VAE

toto
This repository contains the implementation of our Heterogeneous Incomplete Variational Autoendoder model (HI-VAE). It has been written in Python, using PyTorch.

The details of this model are included in this [paper](https://arxiv.org/abs/1807.03653). Please cite it if you use this code for your own research.

This implementation also extends HI-VAE to support survival data, enabling the model to handle time-to-event analysis with incomplete and heterogeneous data types.

## Database description

There are three different datasets considered in the experiments (Wine, Adult and Default Credit). Each dataset has each own folder, containing:

* **data.csv**: the dataset
* **data_types.csv**: a csv containing the types of that particular dataset. Every line is a different attribute containing three paramenters:
  	* name: real, pos (positive), cat (categorical), ord (ordinal), count, surv (log-normal) or surv_weibul (Weibull)
   	* type: real, pos (positive), cat (categorical), ord (ordinal), count
	* dim: dimension of the variable in the original dataset
	* nclass: number of categories (for cat and ord)
* **Missingxx_y.csv**: a csv containing the positions of the different missing values in the data. Each "y" mask was generated randomly, containing a "xx" % of missing values. This file may be left blank if no missing values need to be specified.

You can add your own datasets as long as they follow this structure.

## Files description

* **src.py**: Contains the HI-VAE models (factorized encoder or input dropout encoder).
* **utils**: This folder contains different scripts to support load data, compute likelihood, compute error.
* **data_preprocessing __ .ipynb**: Data reprocessing notebooks.
* **tutorial __ .ipynb**: Tutorial notebooks.


# Code Pre-requisites

First,
```console
$ cd HI-VAE
$ conda create --name hivae
$ conda activate hivae
$ pip install -r pip_requirements.txt
$ chmod +x script_HIVAE.sh
```

Then, run preprocessing and tutorial on your data.


To have an environment compatible with Synthcity,

```console
$ cd HI-VAE
$ conda create --name env_synthcity
$ conda activate env_synthcity
$ pip install synthcity
$ pip install -r pip_requirements_synthcity.txt
```

