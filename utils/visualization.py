#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def plot_data(data, feat_types_dict):
    """
    Visualize different data types.

    Args:
    - data (np.ndarray): Input data (shape: [n_samples, n_features]).
    - feat_types_dict (list): List of feature type dictionaries (e.g., [{'type': 'real', 'dim': 1}, ...]).
    """
    num_features = len(feat_types_dict)
    n_cols = num_features // 2 + num_features % 2
    fig, axes = plt.subplots(n_cols, 2, figsize=(18, 3 * num_features))

    feat_idx = 0
    for i, feature in enumerate(feat_types_dict):
        feature_type = feature['type']
        feat_name = "feat_"+ str(i+1)

        ax= axes[i // 2, i % 2]
        if feature_type in ['cat', 'count', 'ordinal']:  # Count, ordinal & categorical data
            feature_data = pd.DataFrame(data[:, feat_idx].int(), columns=[feat_name])
            # sns.countplot(data=feature_data, x=feat_name, hue=feat_name, palette="Set1", alpha=0.8, legend=False, ax=ax)
            sns.countplot(data=feature_data, x=feat_name, hue=feat_name, alpha=0.8, legend=False, ax=ax)
            ax.set_title(f"Count plot of {feat_name} ({feature_type})", fontsize=16, fontweight="bold")
            
            n_class = np.unique(feature_data.values).shape[0]
            if n_class > 20:
                # Dynamically reduce the number of x-ticks
                ticks = ax.get_xticks()  # Get original tick positions
                labels = ax.get_xticklabels()  # Get original labels

                step = max(1, len(labels) // 10)  # Show every 10th label
                reduced_ticks = ticks[::step]
                reduced_labels = labels[::step]

                ax.set_xticks(reduced_ticks)  # Ensure same number of locations
                ax.set_xticklabels([label.get_text() for label in reduced_labels])  # Set labels

        elif feature_type in ["surv"]:
            feature_data = pd.DataFrame(data[:, feat_idx], columns=[feat_name])
            sns.histplot(feature_data, kde=False, color="royalblue", ax=ax)
            ax.set_title(f"Distribution plot of {feat_name} ({feature_type})", fontsize=16, fontweight="bold")
            ax.legend().set_visible(False)
            feat_idx += 1

        else:
            feature_data = pd.DataFrame(data[:, feat_idx], columns=[feat_name])
            sns.histplot(feature_data, kde=False, color="royalblue", ax=ax)
            ax.set_title(f"Distribution plot of {feat_name} ({feature_type})", fontsize=16, fontweight="bold")
            ax.legend().set_visible(False)
        
        feat_idx += 1
        # Enhance visualization
        ax.grid(True)
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=16, fontweight="semibold")

    plt.show()

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
    num_feats = len(types_dict)
    fig, axes = plt.subplots(num_feats, 2, figsize=(20, 4 * num_feats))

    if num_feats == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure subplots remain consistent for a single feature

    feat_idx = 0
    for i, feature in enumerate(types_dict):
        feature_type = feature['type']

        if feature_type in ['surv']:
            # Extract observed and missing data
            true_observed = true_values[true_values[:, feat_idx + 1] == 1, feat_idx].cpu().numpy()
            true_missing = true_values[true_values[:, feat_idx + 1] == 0, feat_idx].cpu().numpy()
            est_observed = estimated_values[true_values[:, feat_idx + 1] == 1, feat_idx].cpu().numpy()
            est_missing = estimated_values[true_values[:, feat_idx + 1] == 0, feat_idx + 1].cpu().numpy()
            feat_idx += 1
        else:
            # Extract observed and missing data
            true_observed = true_values[miss_mask[:, i] == 1, feat_idx].cpu().numpy()
            true_missing = true_values[miss_mask[:, i] == 0, feat_idx].cpu().numpy()
            est_observed = estimated_values[miss_mask[:, i] == 1, feat_idx].cpu().numpy()
            est_missing = estimated_values[miss_mask[:, i] == 0, feat_idx].cpu().numpy()

        # Select a subset of samples for visualization
        num_obs_samples = min(num_sel_samples, len(true_observed))
        num_miss_samples = min(num_sel_samples, len(true_missing))

        obs_indices = np.random.choice(len(true_observed), num_obs_samples, replace=False) if num_obs_samples > 0 else []
        miss_indices = np.random.choice(len(true_missing), num_miss_samples, replace=False) if num_miss_samples > 0 else []

        true_observed_subset, est_observed_subset = true_observed[obs_indices], est_observed[obs_indices]
        true_missing_subset, est_missing_subset = true_missing[miss_indices], est_missing[miss_indices]

        # Plot observed values
        ax_obs = axes[i, 0]
        if feature_type in ['surv']:
            ax_obs.set_title(f"Feature {i+1}: {feature_type} (Survival time)", fontsize=15, fontweight='bold')
        else:
            ax_obs.set_title(f"Feature {i+1}: {feature_type} (Observed)", fontsize=15, fontweight='bold')
        ax_obs.set_xlabel("Sample Index")
        ax_obs.set_ylabel("Value")
        ax_obs.grid(True)

        # Plot missing values
        ax_miss = axes[i, 1]
        if feature_type in ['surv']:
            ax_miss.set_title(f"Feature {i+1}: {feature_type} (Censoring time)", fontsize=15, fontweight='bold')
        else:
            ax_miss.set_title(f"Feature {i+1}: {feature_type} (Missing)", fontsize=15, fontweight='bold')
        ax_miss.set_xlabel("Sample Index")
        ax_miss.set_ylabel("Value")
        ax_miss.grid(True)

        if feature_type in ['real', 'pos', 'cat', 'surv']:  # Continuous & categorical data
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
        feat_idx += 1

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