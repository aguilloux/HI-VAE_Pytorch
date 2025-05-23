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

log.remove()  # Remove default handlers
warnings.filterwarnings("ignore")


'''
    ------------ Load data ------------
'''

dataset_name = "Aids"
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
print(aids_control.head())

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


'''
    ------------ HI-VAE hyperparameters optimization ------------
'''

T_surv = torch.Tensor(df_init_control.time)
T_surv_norm = (T_surv - T_surv.min()) / (T_surv.max() - T_surv.min())
n_intervals = 5
T_intervals = torch.linspace(0., T_surv_norm.max(), n_intervals)
T_intervals = torch.cat([T_intervals, torch.tensor([2 * T_intervals[-1] - T_intervals[-2]])])
intervals = [(T_intervals[i].item(), T_intervals[i + 1].item()) for i in range(len(T_intervals) - 1)]

best_params_hivae, study_hivae = generators_dict["HI-VAE"].optuna_hyperparameter_search((data_init_control_encoded, intervals), 
                                                                                        data_init_control,
                                                                                        miss_mask_control, 
                                                                                        true_miss_mask_control, 
                                                                                        feat_types_file_control, 
                                                                                        feat_types_dict, 
                                                                                        n_generated_sample=50, 
                                                                                        n_splits=5,
                                                                                        n_trials=30, 
                                                                                        columns=aids_control_fnames, 
                                                                                        study_name="optuna_study_aids_HI-VAE",)
best_params_dict["HI-VAE"] = best_params_hivae
study_dict["HI-VAE"] = study_hivae
with open("best_params_aids_HI-VAE.json", "w") as f:
    json.dump(best_params_hivae, f)


'''
    ------------ Surv-VAE hyperparameters optimization ------------
'''

best_params_survae, study_survae = generators_dict["Surv-VAE"].optuna_hyperparameter_search(data_init_control, 
                                                                                            columns=aids_control_fnames, 
                                                                                            target_column="censor", 
                                                                                            time_to_event_column="time", 
                                                                                            n_generated_sample=50, 
                                                                                            n_splits=5,
                                                                                            n_trials=30, 
                                                                                            study_name="optuna_study_aids_Surv-VAE",)
best_params_dict["Surv-VAE"] = best_params_survae
study_dict["Surv-VAE"] = study_survae

with open("best_params_aids_Surv-VAE.json", "w") as f:
    json.dump(best_params_survae, f)


'''
    ------------ Surv-GAN hyperparameters optimization ------------
'''

best_params_survgan, study_survgan = generators_dict["Surv-GAN"].optuna_hyperparameter_search(data_init_control, 
                                                                                            columns=aids_control_fnames, 
                                                                                            target_column="censor", 
                                                                                            time_to_event_column="time", 
                                                                                            n_generated_sample=50, 
                                                                                            n_splits=5,
                                                                                            n_trials=30, 
                                                                                            study_name="optuna_study_aids_Surv-GAN",)
best_params_dict["Surv-GAN"] = best_params_survgan
study_dict["Surv-GAN"] = study_survgan

with open("best_params_aids_Surv-GAN.json", "w") as f:
    json.dump(best_params_survgan, f)
