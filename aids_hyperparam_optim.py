import pandas as pd
import torch
import optuna

from loguru import logger
logger.remove()  # Remove any default sinks set up automatically

from utils import data_processing, visualization

import warnings
warnings.filterwarnings("ignore")

import synthcity.logger as log
import sys
import json
import os

log.remove()  # Remove default handlers
warnings.filterwarnings("ignore")


name_config = "aids_surv_weibull"  # name of the configuration for the optuna study
n_trials = 50 # number of trials for each generator
n_splits = 5 # number of splits for cross-validation
n_generated_samples = 100 # number of generated datasets each time
dataset_name = "Aids"


'''
    ------------ Load data ------------
'''

data_file_control= "./dataset/" + dataset_name + "/data_control.csv"
feat_types_file_control = "./dataset/" + dataset_name + "/data_types_control.csv"
data_file_treated= "./dataset/" + dataset_name + "/data_treated.csv"
feat_types_file_treated= "./dataset/" + dataset_name + "/data_types_treated.csv"

# If the dataset has no missing data, leave the "miss_file" variable empty
m_perc = 10
mask = 1
miss_file = "dataset/" + dataset_name + "/Missing{}_{}.csv".format(m_perc, mask)
true_miss_file = None

aids_control_fnames = ['time', 'censor'] + pd.read_csv(feat_types_file_control)["name"].to_list()[1:]
aids_control = pd.read_csv(data_file_control, header=None, names=aids_control_fnames)

# Load and transform control data
data_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control, feat_types_file_control, miss_file, true_miss_file)
data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

# Load and transform treated data
data_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, miss_file, true_miss_file)
data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)

# Format data in dataframe
df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=aids_control_fnames)
df_init_control = pd.DataFrame(data_init_control.numpy(), columns=aids_control_fnames)

# Update dataframe
df_init_treated["treatment"] = 1
df_init_control["treatment"] = 0
df_init = pd.concat([df_init_control, df_init_treated], ignore_index=True)


'''
    ------------ Load models ------------
'''

from execute import surv_hivae, surv_gan, surv_vae
generators_dict = {"HI-VAE" : surv_hivae, 
                   "Surv-GAN" : surv_gan, 
                   "Surv-VAE" : surv_vae}
best_params_dict = {}
study_dict = {}
generators_sel = ["HI-VAE", "Surv-GAN", "Surv-VAE"]


'''
    ------------ Hyperparameters optimization ------------
'''


best_params_dict, study_dict = {}, {}
for generator_name in generators_sel:
    db_file = "optuna_results/optuna_study_{}_{}.db".format(name_config, generator_name)
    if os.path.exists(db_file):
        print("This optuna study already exists. Please change the name of the study or remove the file to create a new one.")
    else: 
        print("Creating new optuna study for {}...".format(generator_name))
        if generator_name in ["HI-VAE"]:
            
            # # For piecewise Weibull distribution, we need to define intervals 
            # T_surv = torch.Tensor(df_init_control.time)
            # T_surv_norm = (T_surv - T_surv.min()) / (T_surv.max() - T_surv.min())
            # n_intervals = 5
            # T_intervals = torch.linspace(0., T_surv_norm.max(), n_intervals)
            # T_intervals = torch.cat([T_intervals, torch.tensor([2 * T_intervals[-1] - T_intervals[-2]])])
            # intervals = [(T_intervals[i].item(), T_intervals[i + 1].item()) for i in range(len(T_intervals) - 1)]
            
            # No intervals needed for Weibull distribution
            intervals = None  

            best_params, study = generators_dict[generator_name].optuna_hyperparameter_search((data_init_control_encoded, intervals), 
                                                                                            data_init_control,
                                                                                            miss_mask_control, 
                                                                                            true_miss_mask_control, 
                                                                                            feat_types_file_control, 
                                                                                            feat_types_dict, 
                                                                                            n_generated_sample=n_generated_samples, 
                                                                                            n_splits=n_splits,
                                                                                            n_trials=n_trials, 
                                                                                            columns=aids_control_fnames, 
                                                                                            study_name="optuna_results/optuna_study_{}_{}".format(name_config, generator_name),)
            best_params_dict[generator_name] = best_params
            study_dict[generator_name] = study
            with open("optuna_results/best_params_{}_{}.json".format(name_config, generator_name), "w") as f:
                json.dump(best_params, f)
        else: 
            best_params, study = generators_dict[generator_name].optuna_hyperparameter_search(data_init_control, 
                                                                                            columns=aids_control_fnames, 
                                                                                            target_column="censor", 
                                                                                            time_to_event_column="time", 
                                                                                            n_generated_sample=n_generated_samples, 
                                                                                            n_splits=n_splits,
                                                                                            n_trials=n_trials, 
                                                                                            study_name="optuna_results/optuna_study_{}_{}".format(name_config, generator_name),)
            best_params_dict[generator_name] = best_params
            study_dict[generator_name] = study
            with open("optuna_results/best_params_{}_{}.json".format(name_config, generator_name), "w") as f:
                json.dump(best_params, f)




