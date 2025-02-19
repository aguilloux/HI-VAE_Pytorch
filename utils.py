#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import argparse
import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical, Poisson
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def get_args(argv = None):
    parser = argparse.ArgumentParser(description='Default parameters of the models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=200, help='Size of the batches')
    parser.add_argument('--epochs',type=int,default=5001, help='Number of epochs of the simulations')
    parser.add_argument('--perp',type=int,default=10, help='Perplexity for the t-SNE')
    parser.add_argument('--train', type=int,default=1, help='Training model flag')
    parser.add_argument('--display', type=int,default=1, help='Display option flag')
    parser.add_argument('--save', type=int,default=1000, help='Save variables every save iterations')
    parser.add_argument('--restore', type=int,default=0, help='To restore session, to keep training or evaluation') 
    parser.add_argument('--plot', type=int,default=1, help='Plot results flag')
    parser.add_argument('--dim_latent_s',type=int,default=10, help='Dimension of the categorical space')
    parser.add_argument('--dim_latent_z',type=int,default=2, help='Dimension of the Z latent space')
    parser.add_argument('--dim_latent_y',type=int,default=10, help='Dimension of the Y latent space')
    parser.add_argument('--dim_latent_y_partition',type=int, nargs='+', help='Partition of the Y latent space')
    parser.add_argument('--miss_percentage_train',type=float,default=0.0, help='Percentage of missing data in training')
    parser.add_argument('--miss_percentage_test',type=float,default=0.0, help='Percentage of missing data in test')
    parser.add_argument('--model_name', type=str, default='model_new', help='File of the training model')
    parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
    parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
    parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
    parser.add_argument('--miss_file', type=str, default='Missing_test.csv', help='File with the missing indexes mask')
    parser.add_argument('--true_miss_file', type=str, help='File with the missing indexes when there are NaN in the data')
    
    return parser.parse_args(argv)

def read_data(data_file, types_file, miss_file, true_miss_file):
    """
    Reads data from CSV files, handles missing values, and applies necessary transformations.

    Parameters:
    -----------
    data_file : str
        Path to the CSV file containing the dataset.
    
    types_file : str
        Path to the CSV file specifying the data types and dimensions for each feature.
    
    miss_file : str
        Path to the CSV file indicating the missing values in the dataset.
    
    true_miss_file : str or None
        Path to the CSV file containing the true missing value mask, if available.

    Returns:
    --------
    data : torch.Tensor
        Transformed dataset with categorical, ordinal, and continuous values properly encoded.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.
    
    miss_mask : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values.
    
    true_miss_mask : torch.Tensor
        A binary mask indicating the actual missing values, if provided.
    
    n_samples : int
        The number of samples in the dataset.
    """
    
    # Read types of data from types file
    with open(types_file) as f:
        types_dict = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]

    # Read data from input file and convert to PyTorch tensor
    with open(data_file, 'r') as f:
        data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        data = torch.tensor(data, dtype=torch.float32)
    
    # Handle true missing values if provided
    if true_miss_file:
        with open(true_miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = torch.tensor(missing_positions, dtype=torch.long)

        true_miss_mask = torch.ones((data.shape[0], len(types_dict)))
        true_miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0  # CSV indexes start at 1
        
        # Replace NaNs with appropriate default values
        nan_mask = torch.isnan(data)
        data_filler = torch.zeros(data.shape[1], dtype=torch.float32)
        
        for i, dtype in enumerate(types_dict):
            if dtype['type'] in {'cat', 'ordinal'}:
                unique_vals = torch.unique(data[:, i][~nan_mask[:, i]])  # Get unique non-NaN values
                data_filler[i] = unique_vals[0] if len(unique_vals) > 0 else 0  # Fill with first category
            else:
                data_filler[i] = 0.0  # Fill numerical data with 0
        
        data[nan_mask] = data_filler.repeat(data.shape[0], 1)[nan_mask]
    
    else:
        true_miss_mask = torch.ones((data.shape[0], len(types_dict)))  # No effect on data if no file is provided
    
    # Construct processed data matrices
    data_complete = []
    
    for i, dtype in enumerate(types_dict):
        if dtype['type'] == 'cat':
            # One-hot encoding for categorical data
            cat_data = data[:, i].to(torch.int64)
            unique_vals, indexes = torch.unique(cat_data, return_inverse=True)
            new_categories = torch.arange(int(dtype['dim']), dtype=torch.int64)
            mapped_categories = new_categories[indexes]
            
            one_hot = torch.zeros((data.shape[0], len(new_categories)))
            one_hot[torch.arange(data.shape[0]), mapped_categories] = 1
            data_complete.append(one_hot)
        
        elif dtype['type'] == 'ordinal':
            # Thermometer encoding for ordinal data
            ordinal_data = data[:, i].to(torch.int64)
            unique_vals, indexes = torch.unique(ordinal_data, return_inverse=True)
            new_categories = torch.arange(int(dtype['dim']), dtype=torch.int64)
            mapped_categories = new_categories[indexes]
            
            thermometer = torch.zeros((data.shape[0], len(new_categories) + 1))
            thermometer[:, 0] = 1
            thermometer[torch.arange(data.shape[0]), 1 + mapped_categories] = -1
            thermometer = torch.cumsum(thermometer, dim=1)

            data_complete.append(thermometer[:, :-1])  # Exclude last column
        
        elif dtype['type'] == 'count':
            # Shift zero-based counts if necessary
            count_data = data[:, i].unsqueeze(1)
            if torch.min(count_data) == 0:
                count_data += 1
            data_complete.append(count_data)
        
        else:
            # Keep continuous data as is
            data_complete.append(data[:, i].unsqueeze(1))
    
    # Concatenate processed features
    data = torch.cat(data_complete, dim=1)

    # Read missing mask file
    n_samples, n_variables = data.shape[0], len(types_dict)
    miss_mask = torch.ones((n_samples, n_variables))

    if os.path.isfile(miss_file):
        with open(miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = torch.tensor(missing_positions, dtype=torch.long)
        miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0  # CSV indexes start at 1
    
    return data, types_dict, miss_mask, true_miss_mask, n_samples



def next_batch(data, types_dict, miss_mask, batch_size, index_batch):
    """
    Generates the next minibatch of data and splits it into its respective features.

    Parameters:
    -----------
    data : torch.Tensor
        The complete dataset from which to extract a batch.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.
    
    miss_mask : torch.Tensor
        A binary mask indicating missing values (1 = observed, 0 = missing).
    
    batch_size : int
        The number of samples to include in each batch.
    
    index_batch : int
        The index of the current batch to extract.

    Returns:
    --------
    data_list : list of torch.Tensors
        A list containing feature-wise separated data for the current batch.
    
    miss_list : torch.Tensor
        The corresponding missing data mask for the current batch.
    """
    
    # Extract minibatch
    batch_xs = data[index_batch * batch_size : (index_batch + 1) * batch_size, :]
    
    # Split variables in the batch
    data_list, initial_index = [], 0
    for d in types_dict:
        dim = int(d['dim'])
        data_list.append(batch_xs[:, initial_index : initial_index + dim])
        initial_index += dim
    
    # Extract missing mask for the batch
    miss_list = miss_mask[index_batch * batch_size : (index_batch + 1) * batch_size, :]
    
    return data_list, miss_list

def load_data_types(types_file):
    """
    Reads the types of data from a CSV file and returns a dictionary.

    Parameters:
    -----------
    types_file : str
        Path to the CSV file containing variable types.

    Returns:
    --------
    list of dict:
        A list where each dictionary specifies the type of a variable.
    """
    with open(types_file, newline='') as f:
        return [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]


def batch_normalization(batch_data_list, feat_types_list, miss_list):
    """
    Normalizes real-valued data while leaving categorical/ordinal variables unchanged.

    Parameters:
    -----------
    batch_data_list : list of torch.Tensor
        List of input data tensors, each corresponding to a feature.
    
    feat_types_list : list of dict
        List specifying the type of each feature.
    
    miss_list : torch.Tensor
        Binary mask indicating observed (1) and missing (0) values.

    Returns:
    --------
    normalized_data : list of torch.Tensor
        List of normalized feature tensors.
    
    normalization_parameters : list of tuples
        Normalization parameters for each feature.
    """

    normalized_data = []
    normalization_parameters = []

    for i, d in enumerate(batch_data_list):
        missing_mask = miss_list[:, i] == 0  # True for missing values, False for observed values
        observed_data = d[~missing_mask]  # Extract observed values

        feature_type = feat_types_list[i]['type']

        if feature_type == 'real':
            # Standard normalization (mean 0, std 1)
            data_var, data_mean = torch.var_mean(observed_data, unbiased=False)
            data_var = torch.clamp(data_var, min=1e-6, max=1e20)  # Prevent division by zero
            
            normalized_observed = (observed_data - data_mean) / torch.sqrt(data_var)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask] = normalized_observed  # Assign transformed values
            normalized_d[missing_mask] = 0  # Missing values set to 0
            
            normalization_parameters.append((data_mean, data_var))

        elif feature_type == 'pos':
            # Log-normal transformation and normalization
            observed_data_log = torch.log1p(observed_data)
            data_var_log, data_mean_log = torch.var_mean(observed_data_log, unbiased=False)
            data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)

            normalized_observed = (observed_data_log - data_mean_log) / torch.sqrt(data_var_log)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask] = normalized_observed
            normalized_d[missing_mask] = 0

            normalization_parameters.append((data_mean_log, data_var_log))

        elif feature_type == 'count':
            # Log transformation (No variance normalization)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask] = torch.log1p(observed_data)  # Log-transform observed values
            normalized_d[missing_mask] = 0  # Missing values set to 0
            
            normalization_parameters.append((0.0, 1.0))

        else:
            # Keep categorical and ordinal values unchanged
            normalized_d = d.clone()
            normalization_parameters.append((0.0, 1.0))

        normalized_data.append(normalized_d)

    return normalized_data, normalization_parameters


def s_proposal_multinomial(X, s_layer, tau):
    """
    Proposes a categorical distribution for `s` using the Gumbel-Softmax trick.

    Parameters:
    -----------
    X : torch.Tensor
        Input feature tensor with shape `(batch_size, input_dim)`.
    
    self.s_layer : 
    
    tau : float
        Temperature parameter for Gumbel-Softmax.

    Returns:
    --------
    samples_s : torch.Tensor
        Sampled categorical latent variables using the Gumbel-Softmax trick.
    
    log_pi_aux : torch.Tensor
        Logits of the categorical distribution.
    """
    
    log_pi = s_layer(X)
    log_pi_aux = torch.log_softmax(log_pi, dim=-1)

    gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_pi_aux)))
    samples_s = F.softmax((log_pi_aux + gumbel_noise) / tau, dim=-1)

    return samples_s, log_pi_aux

def z_prior_GMM(samples_s, z_distribution_layer):
    """
    Computes the Gaussian Mixture Model (GMM) prior for `z`.

    Parameters:
    -----------
    samples_s : torch.Tensor
        Sampled categorical latent variables `s`.
    
    z_distribution_layer : 

    Returns:
    --------
    mean_pz : torch.Tensor
        Mean of the Gaussian Mixture Model.
    
    log_var_pz : torch.Tensor
        Log variance (fixed to zero for standard normal prior).
    """
    
    mean_pz = z_distribution_layer(samples_s)
    log_var_pz = torch.zeros_like(mean_pz).clamp(min=-15.0, max=15.0)

    return mean_pz, log_var_pz


def z_proposal_GMM(X, samples_s, batch_size, z_dim, z_layer):
    """
    Proposes a Gaussian Mixture Model (GMM) for latent variable `z`.

    Parameters:
    -----------
    X : torch.Tensor
        Input feature tensor of shape `(batch_size, feature_dim)`.
    
    samples_s : torch.Tensor
        Sampled categorical latent variables of shape `(batch_size, s_dim)`.
    
    batch_size : int
        Number of samples in a batch.
    
    z_dim : int
        Dimensionality of the latent space `z`.

    z_layer :

    Returns:
    --------
    samples_z : torch.Tensor
        Sampled latent variables.
    
    list : [mean_qz, log_var_qz]
        - `mean_qz`: Mean of the latent `z` distribution.
        - `log_var_qz`: Log variance of the latent `z` distribution.
    """

    # Concatenate inputs
    concat_input = torch.cat([X, samples_s], dim=1)

    # Compute mean and log variance
    mean_qz, log_var_qz = torch.chunk(z_layer(concat_input), 2, dim=1) 

    # Avoid numerical instability
    log_var_qz = torch.clamp(log_var_qz, -15.0, 15.0)

    # Reparameterization trick
    eps = torch.randn((batch_size, z_dim), device=X.device)
    samples_z = mean_qz + torch.exp(0.5 * log_var_qz) * eps

    return samples_z, [mean_qz, log_var_qz]


def y_partition(samples_y, feat_types_list, y_dim_partition):
    """
    Partitions `samples_y` according to `y_dim_partition`.

    Parameters:
    -----------
    samples_y : torch.Tensor
        The latent variable `y` tensor of shape `(batch_size, sum(y_dim_partition))`.
    
    feat_types_list : list of dict
        List of dictionaries defining variable types and dimensions.
    
    y_dim_partition : list of int
        List specifying partition sizes for `y`.

    Returns:
    --------
    list of torch.Tensor :
        A list where each entry corresponds to a partitioned segment of `samples_y`.
    """
    
    partition_indices = np.insert(np.cumsum(y_dim_partition), 0, 0)
    
    return [samples_y[:, partition_indices[i]:partition_indices[i+1]] for i in range(len(feat_types_list))]


def theta_estimation_from_ys(samples_y, samples_s, feat_types_list, miss_list, theta_layer):
    """
    Estimates parameters (theta) for each feature type from `samples_y` and `samples_s`.

    Parameters:
    -----------
    samples_y : list of torch.Tensor
        List of partitioned `y` samples, where each entry corresponds to a feature.
    
    samples_s : torch.Tensor
        The latent state variable `s` tensor.
    
    feat_types_list : list of dict
        List specifying feature types and dimensions.
    
    miss_list : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values.

    theta_layer :


    Returns:
    --------
    list :
        A list of estimated parameters (θ) for each feature.
    """
    
    # Mapping feature types to corresponding theta functions
    theta_functions = {
        "real": theta_real,
        "pos": theta_pos,
        "count": theta_count,
        "cat": theta_cat,
        "ordinal": theta_ordinal
    }

    theta = []

    # Compute θ(x_d | y_d) for each feature type
    for i, y_sample in enumerate(samples_y):
        feature_type = feat_types_list[i]['type']

        # Partition the data into observed and missing based on mask
        mask = miss_list[:, i].bool()
        observed_y, missing_y = y_sample[mask], y_sample[~mask]
        observed_s, missing_s = samples_s[mask], samples_s[~mask]
        condition_indices = [~mask, mask]

        # Compute the corresponding theta function
        params = theta_functions[feature_type](observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer["feat_" + str(i)])
        theta.append(params)

    return theta



def theta_real(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the mean and variance layers for real-valued data.

    This function estimates parameters for continuous real-valued features in a survival analysis model.

    Parameters:
    -----------
    observed_y : torch.Tensor
        Tensor of observed `y` values with shape `(batch_size, feature_dim)`.
    
    missing_y : torch.Tensor
        Tensor of missing `y` values with shape `(batch_size, feature_dim)`.
    
    observed_s : torch.Tensor
        Tensor of observed latent states `s` with shape `(batch_size, latent_dim)`.
    
    missing_s : torch.Tensor
        Tensor of missing latent states `s` with shape `(batch_size, latent_dim)`.
    
    condition_indices : list of lists
        Indices for observed and missing data.

    theta_layer : 

    Returns:
    --------
    list :
        `[h2_mean, h2_sigma]` where:
        - `h2_mean` is the estimated mean layer.
        - `h2_sigma` is the estimated variance layer.

    Notes:
    ------
    - This function uses `observed_data_layer` to apply a shared transformation to both observed and missing data.
    """

    # Mean layer
    h2_mean = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean"]
    )

    # Sigma layer
    h2_sigma = observed_data_layer(
        observed_s,
        missing_s,
        condition_indices,
        layer=theta_layer["sigma"]
    )

    return [h2_mean, h2_sigma]



def theta_pos(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the mean and variance layers for positive real-valued data.

    This function estimates parameters for positive real-valued features in a survival analysis model.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_mean, h2_sigma]` where:
        - `h2_mean` is the estimated mean layer.
        - `h2_sigma` is the estimated variance layer.

    Notes:
    ------
    - Identical to `theta_real`, but tailored for **positive** real-valued data.
    """

    # Mean layer
    h2_mean = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean"]
    )

    # Sigma layer
    h2_sigma = observed_data_layer(
        observed_s,
        missing_s,
        condition_indices,
        layer=theta_layer["sigma"]
    )

    return [h2_mean, h2_sigma]


def theta_count(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the lambda layer for count-valued data.

    This function estimates the rate parameter (lambda) for Poisson-distributed count data.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    torch.Tensor :
        The estimated lambda layer (`h2_lambda`).

    Notes:
    ------
    - Used for modeling **count-based** survival features.
    - Applies a **linear transformation** using `observed_data_layer` to compute `lambda`.
    """

    h2_lambda = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer
    )

    return h2_lambda


def theta_cat(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the log-probability layer for categorical data.

    This function estimates log-probabilities for categorical features using a linear layer.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    torch.Tensor :
        Log-probability tensor (`h2_log_pi`) with shape `(batch_size, num_classes)`.

    Notes:
    ------
    - Uses `observed_data_layer` to compute logits for **all but one** class.
    - Ensures **identifiability** by appending a **zero log-probability** for the first category.
    """
    
    h2_log_pi_partial = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer
    )

    # Ensure the first value is zero for identifiability
    h2_log_pi = torch.cat([torch.zeros((h2_log_pi_partial.shape[0], 1)), h2_log_pi_partial], dim=1)

    return h2_log_pi


def theta_ordinal(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the partitioning and mean layers for ordinal data.

    This function estimates parameters for ordinal features using a cumulative distribution approach.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_theta, h2_mean]` where:
        - `h2_theta` represents the estimated partitioning layer.
        - `h2_mean` is the estimated mean layer.

    Notes:
    ------
    - `h2_theta` defines the **ordered category partitions**.
    - `h2_mean` estimates **underlying latent scores** for ordinal regression.
    """

    # Theta layer
    h2_theta = observed_data_layer(
        observed_s,
        missing_s,
        condition_indices,
        layer=theta_layer["theta"]
    )

    # Mean layer
    h2_mean = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean"]
    )

    return [h2_theta, h2_mean]


def observed_data_layer(observed_data, missing_data, condition_indices, layer=None):
    """
    Train a layer with the observed data and reuse it for the missing data in PyTorch.

    Parameters:
    -----------
    observed_data : torch.Tensor
        A tensor containing the observed (non-missing) data.
    
    missing_data : torch.Tensor
        A tensor containing the missing data.
    
    condition_indices : list of lists
        A list containing two lists:
        - `condition_indices[0]`: Indices corresponding to missing data.
        - `condition_indices[1]`: Indices corresponding to observed data.

    layer : 

    Returns:
    --------
    torch.Tensor
        A tensor combining both observed and missing data outputs after transformation.

    """

    # Forward pass for observed data
    obs_output = layer(observed_data)

    # Forward pass for missing data (using same layer but without updates)
    with torch.no_grad():
        miss_output = layer(missing_data)

    # Combine outputs based on condition indices
    output = torch.empty_like(torch.cat([miss_output, obs_output], dim=0))
    output[condition_indices[0]] = miss_output  # Missing data indices
    output[condition_indices[1]] = obs_output   # Observed data indices

    return output


def loglik_evaluation(batch_data_list, feat_types_list, miss_list, theta, normalization_params):
    """
    Evaluates the log-likelihood of observed and missing data.

    Parameters:
    -----------
    batch_data_list : list of torch.Tensor
        List of tensors containing batch-wise feature data.
    
    feat_types_list : list of dict
        List of dictionaries specifying the type of each feature. Each dictionary should contain:
        - 'type': The type of the feature (e.g., 'real', 'cat', 'ordinal', etc.).
    
    miss_list : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values in the dataset.
    
    theta : list of torch.Tensor
        List of parameter tensors associated with each feature for likelihood computation.
    
    normalization_params : list of dict
        List of normalization parameters for each feature, used in likelihood calculations.

    Returns:
    --------
    params_x : list of torch.Tensor
        List of estimated parameters for each feature.
    
    log_p_x : torch.Tensor
        Stacked log-likelihood values for observed data across all features.
    
    log_p_x_missing : torch.Tensor
        Stacked log-likelihood values for missing data across all features (used for test log-likelihood evaluation).
    
    samples_x : list of torch.Tensor
        List of sampled values for each feature from the estimated distributions.

    Notes:
    ------
    - The function dynamically calls the corresponding log-likelihood function from `loglik_models_missing_normalize`.
    - It supports various feature types by using `getattr()` to retrieve the appropriate function dynamically.
    """

    # Initialize storage lists
    params_x, log_p_x, log_p_x_missing, samples_x = [], [], [], []

    loglik_functions = {
        "real": loglik_real,
        "pos": loglik_pos,
        "count": loglik_count,
        "cat": loglik_cat,
        "ordinal": loglik_ordinal
    }

    # Compute log-likelihood for each feature type
    for feature_idx, batch_data in enumerate(batch_data_list):
        feature_type = feat_types_list[feature_idx]['type']
        
        # Dynamically select the appropriate likelihood function
        loglik_function = loglik_functions[feature_type]

        # Prepare input for likelihood function
        batch_data_ext = [batch_data, miss_list[:, feature_idx]]

        # Compute likelihood
        out = loglik_function(batch_data_ext, feat_types_list[feature_idx], theta[feature_idx], normalization_params[feature_idx])

        # Store computed values
        params_x.append(out['params'])
        log_p_x.append(out['log_p_x'])
        log_p_x_missing.append(out['log_p_x_missing'])  # Test log-likelihood
        samples_x.append(out['samples'])

    # Stack log-likelihood tensors for efficient computation
    return params_x, torch.stack(log_p_x), torch.stack(log_p_x_missing), samples_x



def loglik_real(batch_data, list_type, theta, normalization_params):
    """
    Computes the log-likelihood for real-valued (continuous) data under a Gaussian assumption.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed data tensor.
        - `missing_mask`: Binary mask tensor (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : tuple of (torch.Tensor, torch.Tensor)
        - `est_mean`: Predicted mean from the model.
        - `est_var`: Predicted variance from the model.
    
    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - `data_mean`: Mean used for normalizing the original dataset.
        - `data_var`: Variance used for normalizing the original dataset.

    Returns:
    --------
    output : dict
        Dictionary containing:
        - `params`: Estimated mean and variance.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Samples drawn from the estimated distribution.

    Notes:
    ------
    - The function ensures numerical stability using `torch.clamp()`.
    - A Gaussian distribution is assumed for the likelihood function.
    """

    # Extract data and missing mask
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()
    epsilon = 1e-3  # Small constant for numerical stability

    # Retrieve normalization parameters and ensure stability
    data_mean, data_var = normalization_params
    data_var = torch.clamp(data_var, min=epsilon)

    # Retrieve predicted mean and variance, ensuring variance is positive
    est_mean, est_var = theta
    est_var = torch.clamp(F.softplus(est_var), min=epsilon, max=1e20)

    # Perform affine transformation of the parameters
    est_mean = torch.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var

    # Compute log-likelihood using the Gaussian log-likelihood formula
    dim = int(list_type['dim'])
    log_normalization = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi))
    log_variance_term = -0.5 * torch.sum(torch.log(est_var), dim=1)
    log_exponent = -0.5 * torch.sum((data - est_mean) ** 2 / est_var, dim=1)

    log_p_x = log_exponent + log_variance_term + log_normalization

    # Generate output dictionary
    output = {
        "params": [est_mean, est_var],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": Normal(est_mean, torch.sqrt(est_var)).sample()
    }

    return output


def loglik_pos(batch_data, list_type, theta, normalization_params):
    """
    Computes the log-likelihood for positive real-valued data using a log-normal distribution.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed positive real-valued data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : tuple of (torch.Tensor, torch.Tensor)
        - `est_mean`: Predicted log-mean.
        - `est_var`: Predicted log-variance.
    
    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - `data_mean_log`: Log mean of the dataset.
        - `data_var_log`: Log variance of the dataset.

    Returns:
    --------
    output : dict
        - `params`: Estimated mean and variance.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled values from the estimated log-normal distribution.
    """
    epsilon = 1e-3

    # Extract normalization parameters
    data_mean_log, data_var_log = normalization_params
    data_var_log = torch.clamp(data_var_log, min=epsilon)

    # Extract data and mask
    data, missing_mask = batch_data
    data_log = torch.log1p(data)  # Log transform
    missing_mask = missing_mask.float()

    # Extract predicted parameters and ensure positivity
    est_mean, est_var = theta
    est_var = torch.clamp(F.softplus(est_var), min=epsilon, max=1.0)

    # Affine transformation of the parameters
    est_mean = torch.sqrt(data_var_log) * est_mean + data_mean_log
    est_var = data_var_log * est_var

    # Compute log-likelihood
    log_p_x = -0.5 * torch.sum((data_log - est_mean) ** 2 / est_var, dim=1) \
              - 0.5 * torch.sum(torch.log(2 * torch.pi * est_var), dim=1) \
              - torch.sum(data_log, dim=1)

    return {
        "params": [est_mean, est_var],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": torch.clamp(torch.exp(Normal(est_mean, torch.sqrt(est_var)).sample()) - 1.0, min=0, max=1e20)
    }


def loglik_cat(batch_data, list_type, theta, normalization_params):
    """
    Computes the log-likelihood for categorical data.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: One-hot encoded categorical data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : torch.Tensor
        Logits for categorical distribution.

    Returns:
    --------
    output : dict
        - `params`: Logits for categorical distribution.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled categorical values.
    """
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    # Compute log-likelihood
    log_p_x = -F.cross_entropy(theta, data.argmax(dim=1), reduction='none')

    return {
        "params": theta,
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": F.one_hot(Categorical(logits=theta).sample(), num_classes=int(list_type["dim"]))
    }


def loglik_ordinal(batch_data, list_type, theta, normalization_params):
    """
    Computes the log-likelihood for ordinal data using a cumulative distribution approach.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Ordinal data encoded as binary indicators.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : tuple of (torch.Tensor, torch.Tensor)
        - `partition_param`: Parameters defining ordinal category partitions.
        - `mean_param`: Mean parameter for ordinal variable.

    Returns:
    --------
    output : dict
        - `params`: Estimated probabilities for each ordinal category.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled ordinal values.
    """
    epsilon = 1e-6

    data, missing_mask = batch_data
    missing_mask = missing_mask.float()
    batch_size = data.shape[0]

    # Ensure increasing outputs for ordinal categories
    partition_param, mean_param = theta
    mean_value = mean_param.view(-1, 1)
    theta_values = torch.cumsum(torch.clamp(F.softplus(partition_param), min=epsilon, max=1e20), dim=1)

    sigmoid_est_mean = torch.sigmoid(theta_values - mean_value)
    mean_probs = torch.cat([sigmoid_est_mean, torch.ones((batch_size, 1))], dim=1) - \
                 torch.cat([torch.zeros((batch_size, 1)), sigmoid_est_mean], dim=1)

    mean_probs = torch.clamp(mean_probs, min=epsilon, max=1.0)

    # Compute log-likelihood
    true_values = F.one_hot(data.sum(dim=1).long() - 1, num_classes=int(list_type["dim"]))
    log_p_x = -F.cross_entropy(torch.log(mean_probs), true_values.argmax(dim=1), reduction="none")

    # Generate samples from the ordinal distribution
    sampled_values = torch.distributions.Categorical(logits=mean_probs.log()).sample()
    samples = (torch.arange(int(list_type['dim']), device=sampled_values.device)
                         .unsqueeze(0) < (sampled_values + 1).unsqueeze(1)).float()
    
    return {
        "params": mean_probs,
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": samples
    }


def loglik_count(batch_data, list_type, theta, normalization_params):
    """
    Computes the log-likelihood for count data using a Poisson distribution.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed count data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : torch.Tensor
        Predicted Poisson rate (lambda).

    Returns:
    --------
    output : dict
        - `params`: Poisson rate (lambda).
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled count values.
    """
    epsilon = 1e-6

    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    # Ensure lambda is positive
    est_lambda = torch.clamp(F.softplus(theta), min=epsilon, max=1e20)

    # Compute log-likelihood
    log_p_x = -torch.sum(F.poisson_nll_loss(torch.log(est_lambda), data, full=False, reduction='none'), dim=1)

    return {
        "params": est_lambda,
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": Poisson(est_lambda).sample()
    }


def samples_concatenation(samples):
    """
    Concatenates multiple sample batches into a single dataset.

    Parameters:
    -----------
    samples : list of dict
        A list where each element is a dictionary containing batch-wise data with keys:
        - 'x': Feature data
        - 'y': Labels or target values
        - 'z': Latent variables
        - 's': Any additional variables

    Returns:
    --------
    samples_s : torch.Tensor
        Concatenated additional variables across batches.
    samples_z : torch.Tensor
        Concatenated latent variables across batches.
    samples_y : torch.Tensor
        Concatenated labels across batches.
    samples_x : torch.Tensor
        Concatenated feature data across batches.
    """
    
    samples_x = torch.cat([torch.cat(batch['x'], dim=1) for batch in samples], dim=0)
    samples_y = torch.cat([batch['y'] for batch in samples], dim=0)
    samples_z = torch.cat([batch['z'] for batch in samples], dim=0)
    samples_s = torch.cat([batch['s'] for batch in samples], dim=0)
    
    return samples_s, samples_z, samples_y, samples_x

def discrete_variables_transformation(data, types_dict):
    """
    Transforms categorical and ordinal variables into their correct numerical representations.

    Parameters:
    -----------
    data : torch.Tensor
        The dataset containing mixed-type features.
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    torch.Tensor
        A tensor where categorical variables are mapped to their indices,
        and ordinal variables are transformed using sum-based encoding.
    """
    
    ind_ini, output = 0, []
    for d in types_dict:
        ind_end = ind_ini + int(d['dim'])
        subset = data[:, ind_ini:ind_end]  # Extract relevant columns

        if d['type'] == 'cat':
            output.append(torch.argmax(subset, dim=1, keepdim=True))  # Argmax for categorical variables
        elif d['type'] == 'ordinal':
            output.append((torch.sum(subset, dim=1, keepdim=True) - 1))  # Sum-based transformation for ordinal variables
        else:
            output.append(subset)  # Keep continuous variables unchanged
        
        ind_ini = ind_end
    
    return torch.cat(output, dim=1)


def mean_imputation(train_data, miss_mask, types_dict):
    """
    Performs mean and mode imputation for missing values in categorical, ordinal, and continuous data.

    Parameters:
    -----------
    train_data : torch.Tensor
        The dataset containing missing values.
    
    miss_mask : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    torch.Tensor
        The dataset with missing values imputed.
    """
    
    ind_ini, est_data = 0, []
    n_features = len(types_dict)

    for d in range(n_features):
        type_dict = types_dict[d]
        ind_end = ind_ini + (1 if type_dict['type'] in {'cat', 'ordinal'} else int(type_dict['dim']))
        miss_pattern = miss_mask[:, d] == 1  # Extract mask
        
        if type_dict['type'] in {'cat', 'ordinal'}:
            # Mode imputation
            values, counts = torch.unique(train_data[miss_pattern, ind_ini:ind_end], return_counts=True)
            data_mode = values[torch.argmax(counts)]  # Get the mode
        else:
            # Mean imputation
            data_mode = torch.mean(train_data[miss_pattern, ind_ini:ind_end], dim=0)
        
        # Apply imputation
        data_imputed = train_data[:, ind_ini:ind_end] * miss_mask[:, ind_ini:ind_end] + data_mode * (1 - miss_mask[:, ind_ini:ind_end])
        est_data.append(data_imputed)
        
        ind_ini = ind_end
    
    return torch.cat(est_data, dim=1)


def p_distribution_params_concatenation(params, types_dict):
    """
    Concatenates probability distribution parameters from multiple batches for p-distribution.

    Parameters:
    -----------
    params : list of dict
        A list of dictionaries where each dictionary contains distribution parameters for a batch.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    out_dict : dict
        A dictionary containing concatenated probability distribution parameters across all batches.
    """
    
    keys = params[0].keys()
    out_dict = {key: params[0][key] for key in keys}  # Initialize with first batch

    for batch in params[1:]:  # Start from the second batch
        for key in keys:
            if key == 'y':
                out_dict[key] = torch.cat([out_dict[key], batch[key]], dim=0)
            
            elif key == 'z':
                out_dict[key] = (
                    torch.cat([out_dict[key][0], batch[key][0]], dim=0),
                    torch.cat([out_dict[key][1], batch[key][1]], dim=0)
                )

            elif key == 'x':
                for v, attr in enumerate(types_dict):
                    if attr['type'] in {'pos', 'real'}:
                        out_dict[key][v] = (
                            torch.cat([out_dict[key][v][0], batch[key][v][0]], dim=0),
                            torch.cat([out_dict[key][v][1], batch[key][v][1]], dim=0)
                        )
                    else:
                        out_dict[key][v] = torch.cat([out_dict[key][v], batch[key][v]], dim=0)
    
    return out_dict

def q_distribution_params_concatenation(params):
    """
    Concatenates probability distribution parameters from multiple batches for q-distribution.

    Parameters:
    -----------
    params : list of dict
        A list of dictionaries where each dictionary contains distribution parameters for a batch.

    Returns:
    --------
    out_dict : dict
        A dictionary containing concatenated probability distribution parameters across all batches.
    """

    keys = params[0].keys()
    out_dict = {key: params[0][key] if key == 'z' else [params[0][key]] for key in keys}

    for batch in params[1:]:  # Start from the second batch
        for key in keys:
            if key == 'z':
                out_dict[key] = (
                    torch.cat([out_dict[key][0], batch[key][0]], dim=0),
                    torch.cat([out_dict[key][1], batch[key][1]], dim=0)
                )
            else:
                out_dict[key].append(batch[key])

    # Concatenate 's' if it exists in the dictionary
    if 's' in out_dict:
        out_dict['s'] = torch.cat(out_dict['s'], dim=0)

    return out_dict


def statistics(loglik_params, types_dict):
    """
    Computes the mean and mode of various probability distributions based on log-likelihood parameters.

    Parameters:
    -----------
    loglik_params : list of torch.Tensors
        A list containing the log-likelihood parameters for each feature.
    
    types_dict : list of dict
        A list of dictionaries, each specifying the type and dimension of a feature.
        The dictionary should contain a key 'type' which can be:
        - 'real': Continuous real-valued data (assumed normally distributed).
        - 'pos': Positive continuous data (assumed log-normal distributed).
        - 'count': Discrete count data (assumed Poisson distributed).
        - 'cat' or 'ordinal': Categorical or ordinal data.

    Returns:
    --------
    loglik_mean : torch.Tensor
        The mean estimates for each feature based on its respective distribution.
    
    loglik_mode : torch.Tensor
        The mode estimates for each feature based on its respective distribution.
    """

    loglik_mean, loglik_mode = [], []

    for d, attrib in enumerate(loglik_params):
        feature_type = types_dict[d]['type']

        if feature_type == 'real':
            # Normal distribution: mean and mode are the same
            mean, mode = attrib[0], attrib[0]

        elif feature_type == 'pos':
            # Log-normal distribution
            exp_term = torch.exp(attrib[0])
            mean = torch.maximum(exp_term * torch.exp(0.5 * attrib[1]) - 1.0, torch.zeros(1))
            mode = torch.maximum(exp_term * torch.exp(-attrib[1]) - 1.0, torch.zeros(1))

        elif feature_type == 'count':
            # Poisson distribution: mean = lambda, mode = floor(lambda)
            mean, mode = attrib, torch.floor(attrib)

        else:
            # Categorical & ordinal: Mode imputation using argmax
            reshaped_mode = torch.reshape(torch.argmax(attrib, dim=1), (-1, 1))
            mean, mode = reshaped_mode, reshaped_mode
        
        loglik_mean.append(mean)
        loglik_mode.append(mode)

    loglik_mean = torch.squeeze(torch.cat(loglik_mean, dim=1))
    loglik_mode = torch.squeeze(torch.cat(loglik_mode, dim=1))

    return loglik_mean, loglik_mode


def error_computation(x_train, x_hat, types_dict, miss_mask):
    """
    Computes different error metrics (classification error, shift error, and RMSE)
    for observed and missing values based on feature types.

    Parameters:
    -----------
    x_train : torch.Tensor
        The ground truth data (actual values from the dataset).
    
    x_hat : torch.Tensor
        The predicted or imputed values.
    
    types_dict : list of dict
        A list of dictionaries where each dictionary describes a feature's type and dimension.
        The dictionary should contain a key 'type' which can be:
        - 'cat': Categorical data.
        - 'ordinal': Ordinal data.
        - Any other type is treated as continuous (real-valued).
    
    miss_mask : torch.Tensor
        A binary mask indicating missing values (1 = observed, 0 = missing).

    Returns:
    --------
    error_observed : list of torch.Tensors
        A list containing the errors computed on observed values for each feature.
    
    error_missing : list of torch.Tensors
        A list containing the errors computed on missing values for each feature.
    """

    error_observed = []
    error_missing = []
    ind_ini = 0

    for d, feature in enumerate(types_dict):
        feature_type = feature['type']
        dim = int(feature['dim']) if 'dim' in feature else 1  # Default to 1 if 'dim' is not provided
        ind_end = ind_ini + (1 if feature_type in ['cat', 'ordinal'] else dim)

        # Masked values
        observed_mask = miss_mask[:, d] == 1
        missing_mask = miss_mask[:, d] == 0

        x_train_observed = x_train[observed_mask, ind_ini:ind_end]
        x_hat_observed = x_hat[observed_mask, ind_ini:ind_end]
        x_train_missing = x_train[missing_mask, ind_ini:ind_end]
        x_hat_missing = x_hat[missing_mask, ind_ini:ind_end]

        # Classification error (Categorical)
        if feature_type == 'cat':
            error_observed.append(torch.mean((x_train_observed != x_hat_observed).to(torch.float32)))
            error_missing.append(torch.mean((x_train_missing != x_hat_missing).to(torch.float32)) if torch.any(missing_mask) else 0)

        # Shift error (Ordinal)
        elif feature_type == 'ordinal':
            error_observed.append(torch.mean(torch.abs(x_train_observed - x_hat_observed)) / dim)
            error_missing.append(torch.mean(torch.abs(x_train_missing - x_hat_missing)) / dim if torch.any(missing_mask) else 0)

        # Normalized RMSE (Continuous)
        else:
            norm_term = torch.max(x_train[:, d]) - torch.min(x_train[:, d])
            error_observed.append(torch.sqrt(F.mse_loss(x_train_observed, x_hat_observed)) / norm_term)
            error_missing.append(torch.sqrt(F.mse_loss(x_train_missing, x_hat_missing)) / norm_term if torch.any(missing_mask) else 0)

        ind_ini = ind_end  # Move to next feature index

    return torch.Tensor(error_observed), torch.Tensor(error_missing)


def plot_true_vs_estimation(true_values, estimated_values, miss_mask, types_dict, num_sel_samples=100):
    """
    Plots true values vs. estimated values for different data types.

    Args:
    - true_values (torch.Tensor): Ground truth data matrix (shape: [n_samples, n_features]).
    - estimated_values (torch.Tensor): Model-estimated data matrix (shape: [n_samples, n_features]).
    - miss_mask (torch.Tensor): Binary mask (1 = observed, 0 = missing).
    - types_dict (list): List of feature type dictionaries (e.g., [{'type': 'real', 'dim': 1}, ...]).
    - num_sel_samples (int): Number of random samples to plot (default: 100).
    """
    num_features = len(types_dict)
    fig, axes = plt.subplots(num_features, 2, figsize=(20, 4 * num_features))

    if num_features == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure subplots remain consistent for a single feature

    for i, feature in enumerate(types_dict):
        feature_type = feature['type']

        # Extract observed and missing data
        true_observed = true_values[miss_mask[:, i] == 1, i].cpu().numpy()
        true_missing = true_values[miss_mask[:, i] == 0, i].cpu().numpy()
        est_observed = estimated_values[miss_mask[:, i] == 1, i].cpu().numpy()
        est_missing = estimated_values[miss_mask[:, i] == 0, i].cpu().numpy()

        # Select a subset of samples for visualization
        num_obs_samples = min(num_sel_samples, len(true_observed))
        num_miss_samples = min(num_sel_samples, len(true_missing))

        obs_indices = np.random.choice(len(true_observed), num_obs_samples, replace=False) if num_obs_samples > 0 else []
        miss_indices = np.random.choice(len(true_missing), num_miss_samples, replace=False) if num_miss_samples > 0 else []

        true_observed_subset, est_observed_subset = true_observed[obs_indices], est_observed[obs_indices]
        true_missing_subset, est_missing_subset = true_missing[miss_indices], est_missing[miss_indices]

        # Plot observed values
        ax_obs = axes[i, 0]
        ax_obs.set_title(f"Feature {i+1}: {feature_type} (Observed)", fontsize=15, fontweight='bold')
        ax_obs.set_xlabel("Sample Index")
        ax_obs.set_ylabel("Value")
        ax_obs.grid(True)

        # Plot missing values
        ax_miss = axes[i, 1]
        ax_miss.set_title(f"Feature {i+1}: {feature_type} (Missing)", fontsize=15, fontweight='bold')
        ax_miss.set_xlabel("Sample Index")
        ax_miss.set_ylabel("Value")
        ax_miss.grid(True)

        if feature_type in ['real', 'pos', 'cat']:  # Continuous & categorical data
            ax_obs.scatter(range(num_obs_samples), true_observed_subset, label="True", marker='o', alpha=0.6)
            ax_obs.scatter(range(num_obs_samples), est_observed_subset, label="Estimation", marker='x', alpha=0.6)

            ax_miss.scatter(range(num_miss_samples), true_missing_subset, label="True", marker='o', alpha=0.6)
            ax_miss.scatter(range(num_miss_samples), est_missing_subset, label="Estimation", marker='x', alpha=0.6)

        elif feature_type == 'count':  # Count data
            ax_obs.bar(range(num_obs_samples), true_observed_subset, label="True", alpha=0.6)
            ax_obs.bar(range(num_obs_samples), est_observed_subset, label="Estimation", alpha=0.6)

            ax_miss.bar(range(num_miss_samples), true_missing_subset, label="True", alpha=0.6)
            ax_miss.bar(range(num_miss_samples), est_missing_subset, label="Estimation", alpha=0.6)

        elif feature_type == 'ordinal':  # Ordinal (Thermometer Encoding)
            ax_obs.plot(range(num_obs_samples), true_observed_subset, label="True", marker='o')
            ax_obs.plot(range(num_obs_samples), est_observed_subset, label="Estimation", marker='x')

            ax_miss.plot(range(num_miss_samples), true_missing_subset, label="True", marker='o')
            ax_miss.plot(range(num_miss_samples), est_missing_subset, label="Estimation", marker='x')

        ax_obs.legend()
        ax_miss.legend()

    plt.suptitle("True vs. Estimated Values for Different Data Types", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0., 1., .98])  # Adjust for suptitle
    plt.show()


def plot_loss_evolution(loss_track, title, xlabel, ylabel):
    """
    Plot the loss curve

    Parameters
    ----------
    loss_track :  `np.ndarray`, shape=(n_samples, 2)
        Normal array of survival labels

    title : `str`
        Title of the figure

    xlabel : `str`
        Label of x axis

    ylabel : `str`
        Label of y axis
    """
    # plt.figure(figsize=(8, 4))
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(loss_track, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


# Function to print loss metrics
def print_loss(epoch, start_time, ELBO, avg_KL_s, avg_KL_z):
    print("Epoch: [%2d]  time: %4.4f, ELBO_train: %.8f, KL_z: %.8f, KL_s: %.8f, reconstruction loss: %.8f"
          % (epoch, time.time() - start_time, ELBO, avg_KL_z, avg_KL_s, ELBO + avg_KL_z + avg_KL_s))