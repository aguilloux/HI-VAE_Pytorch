#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 19 10:15:49 2025

@author: Van Tuan NGUYEN
"""

import sys
import os
import time
import torch
import torch.optim as optim
import importlib
import utils


def save_model(model, filename):
    """Saves the model state to a file."""
    torch.save(model.state_dict(), filename)
    print(f"Model saved at {filename}")


def load_model(model, filename):
    """Loads model state from a file if it exists."""
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        print("Model restored from", filename)


def train_vae(vae_model, optimizer, train_data, miss_mask, true_miss_mask, types_dict, args):
    """
    Trains the HVAE model.

    Parameters:
    -----------
    vae_model : torch.nn.Module
        The HVAE model to train.

    optimizer : torch.optim.Optimizer
        Optimizer for training.

    train_data : torch.Tensor
        The training dataset.

    miss_mask : torch.Tensor
        Mask for missing values.

    true_miss_mask : torch.Tensor
        True missing mask (ground truth).
    
    types_dict : list
        Data type dictionary.
    
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    start_time = time.time()
    n_batches = train_data.shape[0] // args.batch_size

    for epoch in range(args.epochs):
        avg_loss, avg_KL_s, avg_KL_z = 0.0, 0.0, 0.0
        tau = max(1.0 - 0.01 * epoch, 1e-3)

        # Shuffle dataset
        perm = torch.randperm(train_data.shape[0])
        train_data, miss_mask, true_miss_mask = train_data[perm], miss_mask[perm], true_miss_mask[perm]

        for i in range(n_batches):
            # Get batch data
            data_list, miss_list = utils.next_batch(train_data, types_dict, miss_mask, args.batch_size, i)

            # Compute loss
            optimizer.zero_grad()
            vae_res = vae_model.forward(data_list, miss_list, tau)
            vae_res["neg_ELBO_loss"].backward()
            optimizer.step()

            avg_loss += vae_res["neg_ELBO_loss"].item() / n_batches
            avg_KL_s += vae_res["KL_s"].item() / n_batches
            avg_KL_z += vae_res["KL_z"].mean().item() / n_batches

        if epoch % args.display == 0:
            utils.print_loss(epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)

        # Save model periodically
        if epoch % args.save == 0:
            save_model(vae_model, args.network_file)


def test_vae(vae_model, train_data, miss_mask, types_dict, args):
    """
    Evaluates the trained HVAE model.

    Parameters:
    -----------
    vae_model : torch.nn.Module
        The trained HVAE model.
    
    train_data : torch.Tensor
        The training dataset.
    
    miss_mask : torch.Tensor
        Mask for missing values.
    
    types_dict : list
        Data type dictionary.
    
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    print("Testing...")
    n_batches = train_data.shape[0] // args.batch_size
    avg_loss = 0.0

    with torch.no_grad():
        for i in range(n_batches):
            data_list, miss_list = utils.next_batch(train_data, types_dict, miss_mask, args.batch_size, i)

            vae_res = vae_model.forward(data_list, miss_list, tau=1e-3)
            avg_loss += vae_res["neg_ELBO_loss"].item()

    print(f"Testing complete. Average ELBO: {avg_loss / n_batches:.8f}")


if __name__ == "__main__":
    # Get arguments from parser
    args = utils.get_args(sys.argv[1:])

    # Create directories for saving models
    args.save_dir = f'./saved_networks/{args.save_file}/'
    os.makedirs(args.save_dir, exist_ok=True)
    args.network_file = os.path.join(args.save_dir, f'{args.save_file}.pth')

    print(args)

    # Load training data
    train_data, types_dict, miss_mask, true_miss_mask, n_samples = utils.read_data(
        args.data_file, args.types_file, args.miss_file, args.true_miss_file
    )

    # Adjust batch size if larger than dataset
    args.batch_size = min(args.batch_size, n_samples)

    # Compute real missing mask
    miss_mask = torch.multiply(miss_mask, true_miss_mask)

    # Initialize PyTorch HVAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loading = getattr(importlib.import_module("src"), args.model_name)
    vae_model = model_loading(input_dim=train_data.shape[1], 
                              z_dim=args.dim_latent_z, 
                              y_dim=args.dim_latent_y,
                              s_dim=args.dim_latent_s,
                              y_dim_partition=args.dim_latent_y_partition,
                              feat_types_file=args.types_file).to(device)

    vae_model.device = device

    optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

    # Restore model if needed
    if args.restore:
        load_model(vae_model, args.network_file)

    # Train or test model
    if args.train:
        train_vae(vae_model, optimizer, train_data, miss_mask, true_miss_mask, types_dict, args)
        save_model(vae_model, args.network_file)
    else:
        test_vae(vae_model, train_data, miss_mask, types_dict, args)