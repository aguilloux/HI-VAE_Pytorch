#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Fri Apr 11 14:50:00 2025

@author: Lucas Ducrot
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Poisson
from torchsurv.loss import weibull

import utils.data_processing
import utils.visualization
import utils.statistic

import utils.src
import utils.likelihood
import utils.theta_estimation


import importlib
import os 

import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import warnings
from lifelines import CoxPHFitter
from tableone import TableOne
warnings.filterwarnings("ignore")

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator

from sksurv.util import Surv




def HI_VAE_model(data, miss_mask, true_miss_mask, feat_types_file,feat_types_dict,dataset_name, m_perc, mask,  train_test_share = .9, batch_size = 100, n_generated_sample = 10,model_name="HIVAE_inputDropout", dim_latent_z = 20, dim_latent_y = 15, dim_latent_s = 20, epochs = 1000, lr = 1e-3):
    """
    # Train-test split, definition and optimization of the model on control

    Parameters:
    -----------
    data : list of torch.Tensor
        List of tensors containing batch-wise feature data.
    
    miss_mask : list of dict
        List of dictionaries specifying the type of each feature. Each dictionary should contain:
        - 'type': The type of the feature (e.g., 'real', 'cat', 'ordinal', etc.).
    
    true_miss_mask : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values in the dataset.
    
    feat_types_file : list of torch.Tensor
        List of parameter tensors associated with each feature for likelihood computation.
    
    dataset_name : list of dict
        List of normalization parameters for each feature, used in likelihood calculations.

    m_perc : int
        Number of samples to be generated per an input data point
        
    mask : int
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

    # Train-test split on control
    ##data = data_control
    ##miss_mask = miss_mask_control
    ##true_miss_mask = true_miss_mask_control
    ##feat_types_file = feat_types_file_control
    ##train_test_share = .9
    n_samples = data.shape[0]
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.choice(n_samples, n_train_samples, replace=False)
    test_index = [i for i in np.arange(n_samples) if i not in train_index]
    
    data_train = data[train_index]
    miss_mask_train = miss_mask[train_index]
    true_miss_mask_train = true_miss_mask[train_index]
    
    data_test = data[test_index]
    miss_mask_test = miss_mask[test_index]
    true_miss_mask_test = true_miss_mask[test_index]
    
    # Adjust batch size if larger than dataset
    ##batch_size = 100
    batch_size = min(batch_size, n_train_samples)
    
    # Number of batches
    n_train_samples = data_train.shape[0]
    n_batches_train = int(np.floor(n_train_samples / batch_size))
    n_train_samples = n_batches_train * batch_size
    
    # Compute real missing mask
    miss_mask_train = torch.multiply(miss_mask_train, true_miss_mask_train)
    
    # On test/val
    n_test_samples = data_test.shape[0]
    # Adjust batch size if larger than dataset
    batch_test_size = n_test_samples
    # Number of batches
    n_batches_test = int(np.floor(n_test_samples / batch_test_size))
    
    # Compute real missing mask
    miss_mask_test = torch.multiply(miss_mask_test, true_miss_mask_test)
    ##n_generated_sample = 10

    # model_name="HIVAE_factorized"
    ##model_name="HIVAE_inputDropout"
    ##dim_latent_z = 20
    ##dim_latent_y = 15
    ##dim_latent_s = 20
    ##epochs = 1000
    ##lr = 1e-3
    save_file= "{}_{}_missing_{}_{}_z{}_y{}_s{}_batch_{}".format(model_name, dataset_name, m_perc, mask, dim_latent_z, dim_latent_y, dim_latent_s, batch_size)
    # Create directories for saving models
    save_dir = f'./saved_networks/{save_file}/'
    os.makedirs(save_dir, exist_ok=True)
    network_file = os.path.join(save_dir, f'{save_file}.pth')
    
    # Create PyTorch HVAE model
    model_loading = getattr(importlib.import_module("utils.src"), model_name)
    vae_model = model_loading(input_dim=data_train.shape[1], 
                              z_dim=dim_latent_z, 
                              y_dim=dim_latent_y, 
                              s_dim=dim_latent_s, 
                              y_dim_partition=None, 
                              feat_types_file=feat_types_file)
    
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)



    start_time = time.time()
    loss_train, error_observed_train, error_missing_train = [], [], []
    loss_val, error_observed_val, error_missing_val = [], [], []
    
    rng = np.random.default_rng(seed=42)
    for epoch in range(epochs):
        avg_loss, avg_KL_s, avg_KL_z = 0.0, 0.0, 0.0
        avg_loss_val, avg_KL_s_val, avg_KL_z_val = 0.0, 0.0, 0.0
        samples_list, p_params_list, q_params_list, log_p_x_total, log_p_x_missing_total = [], [], [], [], []
        tau = max(1.0 - 0.01 * epoch, 1e-3)
    
        # Shuffle training data
        perm = rng.permutation(data_train.shape[0])
        data_train = data_train[perm]
        miss_mask_train = miss_mask_train[perm]
        true_miss_mask_train = true_miss_mask_train[perm]
    
        for i in range(n_batches_train):
            # Get batch data
            data_list, miss_list = utils.data_processing.next_batch(data_train, feat_types_dict, miss_mask_train, batch_size, i)
    
            # Mask unknown data (set unobserved values to zero)
            data_list_observed = [data * miss_list[:, i].view(batch_size, 1) for i, data in enumerate(data_list)]
    
    
            # Compute loss
            optimizer.zero_grad()
            vae_res = vae_model.forward(data_list_observed, data_list, miss_list, tau, n_generated_sample=1)
            vae_res["neg_ELBO_loss"].backward()
            optimizer.step()
    
            with torch.no_grad():
    
                avg_loss += vae_res["neg_ELBO_loss"].item() / n_batches_train
                avg_KL_s += torch.mean(vae_res["KL_s"]).item() / n_batches_train
                avg_KL_z += torch.mean(vae_res["KL_z"]).item() / n_batches_train
                # Number of samples generated by one input data
    
                
    
                
                for i in range(n_batches_test):
                    data_list_test, miss_list_test = utils.data_processing.next_batch(data_test, feat_types_dict, miss_mask_test, batch_test_size, i)
                
                    # Mask unknown data (set unobserved values to zero)
                    data_list_observed_test = [data * miss_list_test[:, i].view(batch_test_size, 1) for i, data in enumerate(data_list_test)]
                
                    vae_res_test = vae_model.forward(data_list_observed_test, data_list_test, miss_list_test, tau=1e-3, n_generated_sample=1)
                    avg_loss_val += vae_res_test["neg_ELBO_loss"].item() / n_batches_train
                    avg_KL_s_val += torch.mean(vae_res_test["KL_s"]).item() / n_batches_train
                    avg_KL_z_val += torch.mean(vae_res_test["KL_z"]).item() / n_batches_train
                    #print(avg_loss_val)
                
                
         
    
            # Save the generated samlpes and estimated parameters !
            samples_list.append(vae_res["samples"])
            p_params_list.append(vae_res["p_params"])
            q_params_list.append(vae_res["q_params"])
            log_p_x_total.append(vae_res["log_p_x"])
            log_p_x_missing_total.append(vae_res["log_p_x_missing"])
    
    
        #Concatenate samples in arrays
        s_total, z_total, y_total, est_data_train = utils.statistic.samples_concatenation(samples_list)
        
        # Transform discrete variables back to the original values
        data_train_transformed = utils.data_processing.discrete_variables_transformation(data_train[: n_train_samples], feat_types_dict)
        est_data_train_transformed = utils.data_processing.discrete_variables_transformation(est_data_train[0], feat_types_dict)
        # est_data_train_mean_imputed = statistic.mean_imputation(data_train_transformed, miss_mask_train[: n_train_samples], feat_types_dict)
    
        # Compute errors
        error_observed_samples, error_missing_samples = utils.statistic.error_computation(data_train_transformed, est_data_train_transformed, 
                                                                                    feat_types_dict, miss_mask[:n_train_samples])
        
        # # #Create global dictionary of the distribution parameters
        q_params_complete = utils.statistic.q_distribution_params_concatenation(q_params_list)
        
        #Number of clusters created
        cluster_index = torch.argmax(q_params_complete['s'], 1)
        cluster = torch.unique(cluster_index)
        #print('Clusters: ' + str(len(cluster)))
    
        # Save average loss and error
        loss_train.append(avg_loss)
        loss_val.append(avg_loss_val)
        error_observed_train.append(torch.mean(error_observed_samples))
        error_missing_train.append(torch.mean(error_missing_samples))
        if epoch % 100 == 0:
            utils.visualization.print_loss(epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)
    
    print("Training finished.")
    
    torch.save(vae_model.state_dict(), network_file)
    
    utils.visualization.plot_loss_evolution(-np.array(loss_train), title = "HI_VAE loss over epoch",
                                    xlabel = "Epoch", ylabel = "ELBO")
    utils.visualization.plot_loss_evolution(-np.array(loss_val), title = "HI_VAE loss over epoch val",
                                    xlabel = "Epoch", ylabel = "ELBO")

    return vae_model








def HI_VAE_generation(vae_model,data_forgen, feat_types_dict, miss_mask_forgen,true_miss_mask_forgen,n_batches_generation = 1, n_generated_sample = 100):
    """
    # Train-test split, definition and optimization of the model on control

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
    n_samples_forgen = data_forgen.shape[0]
    batch_size = n_samples_forgen
    print(n_samples_forgen)
    # Compute real missing mask
    miss_mask_forgen = torch.multiply(miss_mask_forgen, true_miss_mask_forgen)
    
    # Number of samples generated by one input data
    
    with torch.no_grad(): 
        
        error_mode_global = 0.0
        avg_loss = 0.0
        samples_list = []
        
        for i in range(n_batches_generation):
            data_list, miss_list = utils.data_processing.next_batch(data_forgen, feat_types_dict, miss_mask_forgen, batch_size, i)
    
            # Mask unknown data (set unobserved values to zero)
            data_list_observed = [data * miss_list[:, i].view(batch_size, 1) for i, data in enumerate(data_list)]
            
            vae_res = vae_model.forward(data_list_observed, data_list, miss_list, tau=1e-3, n_generated_sample=n_generated_sample)
            samples_list.append(vae_res["samples"])
        
        
        #Concatenate samples in arrays
        est_data_gen = utils.statistic.samples_concatenation(samples_list)[-1]
        est_data_gen_transformed = []
        for j in range(n_generated_sample):
            data_trans = utils.data_processing.discrete_variables_transformation(est_data_gen[j], feat_types_dict)
            data_trans = utils.data_processing.survival_variables_transformation(data_trans,feat_types_dict)
            est_data_gen_transformed.append(data_trans.unsqueeze(0))
            
        est_data_gen_transformed = torch.cat(est_data_gen_transformed, dim=0)

    return est_data_gen_transformed
