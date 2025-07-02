import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from utils import data_processing
from utils.simulations import *
from execute import surv_hivae, surv_gan, surv_vae

import os
import json
import optuna

import time

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)


n_samples = 600
n_features_bytype = 6
n_active_features = 3 
treatment_effect = 0.
p_treated = 0.5
shape_T = 2.
shape_C = 2.
scale_C = 2.5
scale_C_indep = 3.9
feature_types_list = ["real", "cat"]
independent = False
data_types_create = True

control, treated, types = simulation(treatment_effect, n_samples, independent, feature_types_list,
                                         n_features_bytype, n_active_features, p_treated, shape_T,
                                         shape_C, scale_C, scale_C_indep, data_types_create, seed=0)

control = control.drop(columns='treatment')
treated = treated.drop(columns='treatment')

dataset_name = "Simulations_6_dep_bis"
data_file_control= "./dataset/" + dataset_name + "/data_control.csv"
feat_types_file_control = "./dataset/" + dataset_name + "/data_types_control.csv"
data_file_treated= "./dataset/" + dataset_name + "/data_treated.csv"
feat_types_file_treated= "./dataset/" + dataset_name + "/data_types_treated.csv"

# If the dataset has no missing data, leave the "miss_file" variable empty
miss_file = "dataset/" + dataset_name + "/Missing.csv"
true_miss_file = None

control.to_csv(data_file_control, index=False, header=False)
types.to_csv(feat_types_file_control)
treated.to_csv(data_file_treated, index=False, header=False)
types.to_csv(feat_types_file_treated)


# Load and transform control data
df_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control, feat_types_file_control, miss_file, true_miss_file)
data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

# Load and transform treated data
df_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, miss_file, true_miss_file)
data_init_treated_encoded = torch.from_numpy(df_init_treated_encoded.values)
data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)

fnames = types['name'][:-1].tolist()
fnames.append("time")#.append("censor")
fnames.append("censor")

# Format data in dataframe
df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)

# Update dataframe
df_init_treated["treatment"] = 1
df_init_control["treatment"] = 0
df_init = pd.concat([df_init_control, df_init_treated], ignore_index=True)

# Parameters of the optuna study
metric_optuna = "survival_km_distance" # metric to optimize in optuna
n_splits = 5 # number of splits for cross-validation
n_generated_dataset = 200 # number of generated datasets per fold to compute the metric
name_config = "simu_N{}_nfeat{}_t{}".format(n_samples, n_features_bytype, int(treatment_effect))


generators_sel = ["HI-VAE_piecewise"]
generators_dict = {"HI-VAE_weibull" : surv_hivae,
                   "HI-VAE_piecewise" : surv_hivae,
                   "Surv-GAN" : surv_gan,
                   "Surv-VAE" : surv_vae, 
                    "HI-VAE_weibull_prior" : surv_hivae, 
                    "HI-VAE_piecewise_prior" : surv_hivae}

original_dir = os.getcwd()

best_params_dict_survkm_local, study_dict_survkm_local = {}, {}
best_params_dict_survkm_cluster, study_dict_survkm_cluster = {}, {}
for generator_name in generators_sel:
    n_trials = 150

    dataset_name = "Simulations_6_dep_bis"

    study_name_cluster = "/Users/pchassat/Documents/HI-VAE_Pytorch/dataset/" + dataset_name + "/optuna_results/optuna_study_{}_ntrials{}_{}_{}".format(name_config, n_trials, metric_optuna, generator_name)
    study_name = "dataset/" + dataset_name + "/optuna_results/optuna_study_{}_ntrials{}_{}_{}".format(name_config, n_trials, metric_optuna, generator_name)
    best_params_file = "dataset/" + dataset_name + "/optuna_results/best_params_{}_ntrials{}_{}_{}.json".format(name_config, n_trials, metric_optuna, generator_name)
    db_file = study_name + ".db"
    if os.path.exists(db_file):
        print("This optuna study already exists. Please change the name of the study or remove the file to create a new one.")
        study = optuna.load_study(study_name=study_name_cluster, storage="sqlite:///{}".format(db_file))
        study_dict_survkm_local[generator_name] = study
        with open(best_params_file, "r") as f:
            best_params_dict_survkm_local[generator_name] = json.load(f)
    else:
        raise FileNotFoundError("The optuna study does not exist. Please create it first.")
    
    dataset_name = "Simulations_6_dep"

    study_name_cluster = "/home/pchassat/HI-VAE_Pytorch/dataset/" + dataset_name + "/optuna_results/optuna_study_{}_ntrials{}_{}_{}".format(name_config, n_trials, metric_optuna, generator_name)
    study_name = "dataset/" + dataset_name + "/optuna_results/optuna_study_{}_ntrials{}_{}_{}".format(name_config, n_trials, metric_optuna, generator_name)
    best_params_file = "dataset/" + dataset_name + "/optuna_results/best_params_{}_ntrials{}_{}_{}.json".format(name_config, n_trials, metric_optuna, generator_name)
    db_file = study_name + ".db"
    if os.path.exists(db_file):
        print("This optuna study already exists. Please change the name of the study or remove the file to create a new one.")
        study = optuna.load_study(study_name=study_name_cluster, storage="sqlite:///{}".format(db_file))
        study_dict_survkm_cluster[generator_name] = study
        with open(best_params_file, "r") as f:
            best_params_dict_survkm_cluster[generator_name] = json.load(f)
    else:
        raise FileNotFoundError("The optuna study does not exist. Please create it first.")
    
# Default parameters
n_generated_dataset = 200
generators_sel = ["HI-VAE_piecewise"]
data_gen_control_dict = {}
for generator_name in generators_sel:
    print("=" * 100)
    print("Generate data by " + generator_name)
    if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
        feat_types_dict_ext = feat_types_dict.copy()
        for i in range(len(feat_types_dict)):
            if feat_types_dict_ext[i]['name'] == "survcens":
                if generator_name in ["HI-VAE_weibull", "HI-VAE_weibull_prior"]:
                    feat_types_dict_ext[i]["type"] = 'surv_weibull'
                elif generator_name in ["HI-VAE_lognormal"]:
                    feat_types_dict_ext[i]["type"] = 'surv'
                else:
                    feat_types_dict_ext[i]["type"] = 'surv_piecewise'
        if generator_name in ["HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            gen_from_prior = True
        else:
            gen_from_prior = False
        print(generator_name)
        data_gen_control_dict[generator_name] = generators_dict[generator_name].run(df_init_control_encoded, miss_mask_control, true_miss_mask_control, feat_types_dict_ext, n_generated_dataset, epochs=10000, gen_from_prior=gen_from_prior)


# Cluster parameters
data_gen_control_dict_best_params_survkm_cluster = {}
for generator_name in generators_sel:
    print("=" * 100)
    print("Generate data by " + generator_name)
    best_params = best_params_dict_survkm_cluster[generator_name]
    if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
        feat_types_dict_ext = feat_types_dict.copy()
        for i in range(len(feat_types_dict)):
            if feat_types_dict_ext[i]['name'] == "survcens":
                if generator_name in ["HI-VAE_weibull", "HI-VAE_weibull_prior"]:
                    feat_types_dict_ext[i]["type"] = 'surv_weibull'
                elif generator_name in ["HI-VAE_lognormal"]:
                    feat_types_dict_ext[i]["type"] = 'surv'
                else:
                    feat_types_dict_ext[i]["type"] = 'surv_piecewise'
        if generator_name in ["HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            gen_from_prior = True
        else:
            gen_from_prior = False
        data_gen_control_dict_best_params_survkm_cluster[generator_name] = generators_dict[generator_name].run(df_init_control_encoded,
                                                                                                miss_mask_control,
                                                                                                true_miss_mask_control, 
                                                                                                feat_types_dict_ext, 
                                                                                                n_generated_dataset, 
                                                                                                params=best_params, 
                                                                                                epochs = 10000,
                                                                                                gen_from_prior=gen_from_prior)
    
# Local parameters
data_gen_control_dict_best_params_survkm_local = {}
for generator_name in generators_sel:
    print("=" * 100)
    print("Generate data by " + generator_name)
    best_params = best_params_dict_survkm_local[generator_name]
    if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
        feat_types_dict_ext = feat_types_dict.copy()
        for i in range(len(feat_types_dict)):
            if feat_types_dict_ext[i]['name'] == "survcens":
                if generator_name in ["HI-VAE_weibull", "HI-VAE_weibull_prior"]:
                    feat_types_dict_ext[i]["type"] = 'surv_weibull'
                elif generator_name in ["HI-VAE_lognormal"]:
                    feat_types_dict_ext[i]["type"] = 'surv'
                else:
                    feat_types_dict_ext[i]["type"] = 'surv_piecewise'
        if generator_name in ["HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            gen_from_prior = True
        else:
            gen_from_prior = False
        data_gen_control_dict_best_params_survkm_local[generator_name] = generators_dict[generator_name].run(df_init_control_encoded,
                                                                                                miss_mask_control,
                                                                                                true_miss_mask_control, 
                                                                                                feat_types_dict_ext, 
                                                                                                n_generated_dataset, 
                                                                                                params=best_params, 
                                                                                                epochs = 10000,
                                                                                                gen_from_prior=gen_from_prior)
 

# Convert generated data into dataframe
df_gen_control_dict_best_params_survkm_cluster = {}
df_gen_control_dict_best_params_survkm_local = {}
df_gen_control_dict_default = {}
df_syn_dict_best_params_survkm_local = {}
df_syn_dict_best_params_survkm_cluster = {}
df_syn_dict_default = {}
for generator_name in generators_sel:
    list_df_gen_control_best_params_survkm_local = []
    list_df_gen_control_best_params_survkm_cluster = []
    list_df_gen_control_default = []
    data_syn_best_params_survkm_cluster = []
    data_syn_best_params_survkm_local = []
    data_syn_default = []
    for j in range(n_generated_dataset):
        df_gen_survkm_control_j = pd.DataFrame(data_gen_control_dict_best_params_survkm_local[generator_name][j].numpy(), columns=fnames)
        df_gen_survkm_control_j['treatment'] = 0
        list_df_gen_control_best_params_survkm_local.append(df_gen_survkm_control_j)
        data_syn_best_params_survkm_local.append(pd.concat([df_init_treated, df_gen_survkm_control_j], ignore_index=True))

        df_gen_survkm_control_j = pd.DataFrame(data_gen_control_dict_best_params_survkm_cluster[generator_name][j].numpy(), columns=fnames)
        df_gen_survkm_control_j['treatment'] = 0
        list_df_gen_control_best_params_survkm_cluster.append(df_gen_survkm_control_j)
        data_syn_best_params_survkm_cluster.append(pd.concat([df_init_treated, df_gen_survkm_control_j], ignore_index=True))

        df_gen_control_j = pd.DataFrame(data_gen_control_dict[generator_name][j].numpy(), columns=fnames)
        df_gen_control_j['treatment'] = 0
        list_df_gen_control_default.append(df_gen_control_j)
        data_syn_default.append(pd.concat([df_init_treated, df_gen_control_j], ignore_index=True))
        
    df_gen_control_dict_best_params_survkm_cluster[generator_name] = list_df_gen_control_best_params_survkm_cluster
    df_gen_control_dict_best_params_survkm_local[generator_name] = list_df_gen_control_best_params_survkm_local
    df_gen_control_dict_default[generator_name] = list_df_gen_control_default
    df_syn_dict_best_params_survkm_cluster[generator_name] = data_syn_best_params_survkm_cluster
    df_syn_dict_best_params_survkm_local[generator_name] = data_syn_best_params_survkm_local
    df_syn_dict_default[generator_name] = data_syn_default

from utils.metrics import general_metrics
general_scores_survkm_cluster = []
general_scores_survkm_local = []
general_scores_default = []
for generator_name in generators_sel:
    general_scores_survkm_cluster.append(general_metrics(df_init_control, df_gen_control_dict_best_params_survkm_cluster[generator_name], generator_name))
    general_scores_survkm_local.append(general_metrics(df_init_control, df_gen_control_dict_best_params_survkm_local[generator_name], generator_name))
    general_scores_default.append(general_metrics(df_init_control, df_gen_control_dict_default[generator_name], generator_name))
general_scores_default_df = pd.concat(general_scores_default)
general_scores_survkm_df_cluster = pd.concat(general_scores_survkm_cluster)
general_scores_survkm_df_local = pd.concat(general_scores_survkm_local)


general_scores_default_df.to_csv("dataset/Simulations_6_dep_bis/general_scores_default_df.csv", index=False)
general_scores_survkm_df_cluster.to_csv("dataset/Simulations_6_dep_bis/general_scores_survkm_df_cluster.csv", index=False)
general_scores_survkm_df_local.to_csv("dataset/Simulations_6_dep_bis/general_scores_survkm_df_local.csv", index=False)
