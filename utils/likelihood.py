#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Poisson


def loglik_evaluation(batch_data_list, feat_types_list, miss_list, theta, normalization_params, n_generated_sample=1):
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

    n_generated_sample : int
        Number of samples to be generated per an input data point

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
        "ordinal": loglik_ordinal,
        "surv": loglik_surv,
    }

    # Compute log-likelihood for each feature type
    for feature_idx, batch_data in enumerate(batch_data_list):
        feature_type = feat_types_list[feature_idx]['type']
        
        # Dynamically select the appropriate likelihood function
        loglik_function = loglik_functions[feature_type]

        # Prepare input for likelihood function
        batch_data_ext = [batch_data, miss_list[:, feature_idx]]

        # Compute likelihood
        out = loglik_function(batch_data_ext, feat_types_list[feature_idx], theta[feature_idx], normalization_params[feature_idx], n_generated_sample)

        # Store computed values
        params_x.append(out['params'])
        log_p_x.append(out['log_p_x'])
        log_p_x_missing.append(out['log_p_x_missing'])  # Test log-likelihood
        samples_x.append(out['samples'])

    # Stack log-likelihood tensors for efficient computation
    return params_x, torch.stack(log_p_x), torch.stack(log_p_x_missing), samples_x



def loglik_real(batch_data, list_type, theta, normalization_params, n_generated_sample):
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

    n_generated_sample : int
        Number of samples to be generated per an input data point

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
        "samples": Normal(est_mean, torch.sqrt(est_var)).sample(sample_shape=(n_generated_sample,))
    }

    return output

def loglik_surv(batch_data, list_type, theta, normalization_params, n_generated_sample):
    """
    Computes the log-likelihood for survival data using a log-normal distribution.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed positive real-valued survival times and censoring indicators.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).

    list_type : dict
        Dictionary specifying feature types (not used in function but kept for compatibility).

    theta : tuple of (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        - `est_mean_T`: Predicted log-mean for survival time.
        - `est_var_T`: Predicted log-variance for survival time.
        - `est_mean_C`: Predicted log-mean for censoring time.
        - `est_var_C`: Predicted log-variance for censoring time.

    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - `data_mean_log`: Log mean of the dataset.
        - `data_var_log`: Log variance of the dataset.

    n_samples : int
        Number of samples to generate per input data point.

    Returns:
    --------
    output : dict
        - `params`: Estimated mean and variance for survival and censoring.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled values from the estimated log-normal distribution.
    """

    epsilon = 1e-5  # Small value for numerical stability

    # Extract normalization parameters
    data_mean_log, data_var_log = normalization_params

    # Extract observed data and mask
    data, missing_mask = batch_data
    T_surv, delta = data[:, 0].unsqueeze(1), data[:, 1]
    data_log = torch.log1p(T_surv)  # Log-transform survival times
    missing_mask = missing_mask.float()

    # Extract predicted parameters and enforce positivity
    est_mean_T, est_var_T, est_mean_C, est_var_C = theta
    est_var_T = F.softplus(est_var_T).clamp(min=epsilon, max=1.0)
    est_var_C = F.softplus(est_var_C).clamp(min=epsilon, max=1.0)

    # Transform estimated parameters to original scale
    sqrt_var_log = torch.sqrt(data_var_log)
    est_mean_T = sqrt_var_log * est_mean_T + data_mean_log
    est_var_T *= data_var_log
    est_mean_C = sqrt_var_log * est_mean_C + data_mean_log
    est_var_C *= data_var_log

    # Compute log-likelihood components
    def log_normal_likelihood(data, mean, var):
        return (-0.5 * (data - mean).pow(2) / var
                - 0.5 * torch.log(2 * torch.pi * var)
                - data)

    log_p_x_T = log_normal_likelihood(data_log, est_mean_T, est_var_T).sum(dim=1)
    log_p_x_C = log_normal_likelihood(data_log, est_mean_C, est_var_C).sum(dim=1)

    # Compute overall log-likelihood based on censoring indicator
    log_p_x = (log_p_x_T * delta + log_p_x_C * (1 - delta))

    # Generate samples from estimated log-normal distributions
    max_threshold = T_surv.max().item()
    def sample_from_log_normal(mean, var):
        return torch.clamp(
            torch.exp(Normal(mean, torch.sqrt(var)).sample((n_generated_sample,))) - 1.0,
            min=0, max=max_threshold
        )

    sample_T = sample_from_log_normal(est_mean_T, est_var_T)
    sample_C = sample_from_log_normal(est_mean_C, est_var_C)

    return {
        "params": [est_mean_T, est_var_T, est_mean_C, est_var_C],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": torch.cat((sample_T, sample_C), dim=-1),
    }

def loglik_pos(batch_data, list_type, theta, normalization_params, n_generated_sample):
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

    n_generated_sample : int
        Number of samples to be generated per an input data point

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
        "samples": torch.clamp(torch.exp(Normal(est_mean, torch.sqrt(est_var)).sample(sample_shape=(n_generated_sample, ))) - 1.0, min=0, max=1e20)
    }


def loglik_cat(batch_data, list_type, theta, normalization_params, n_generated_sample):
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

    n_generated_sample : int
        Number of samples to be generated per an input data point

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
        "samples": F.one_hot(Categorical(logits=theta).sample(sample_shape=(n_generated_sample, )), num_classes=int(list_type["dim"]))
    }


def loglik_ordinal(batch_data, list_type, theta, normalization_params, n_generated_sample):
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

    n_generated_sample : int
        Number of samples to be generated per an input data point

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
    true_values = F.one_hot(data.sum(dim=1).long() - 1, num_classes=int(list_type["nclass"]))
    log_p_x = -F.cross_entropy(torch.log(mean_probs), true_values.argmax(dim=1), reduction="none")

    # Generate samples from the ordinal distribution
    sampled_values = torch.distributions.Categorical(logits=mean_probs.log()).sample(sample_shape=(n_generated_sample, ))
    samples = []
    for i in range(n_generated_sample):
        samples.append((torch.arange(int(list_type["nclass"]), device=sampled_values.device)
                            .unsqueeze(0) < (sampled_values[i] + 1).unsqueeze(1)).float().unsqueeze(0))
    samples = torch.cat(samples, dim=0)
    
    return {
        "params": mean_probs,
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": samples
    }


def loglik_count(batch_data, list_type, theta, normalization_params, n_generated_sample):
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

    n_generated_sample : int
        Number of samples to be generated per an input data point

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
        "samples": Poisson(est_lambda).sample(sample_shape=(n_generated_sample, ))
    }