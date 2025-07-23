import numpy as np
import pandas as pd
import torch
from scipy.linalg import toeplitz
from scipy.stats import norm

import matplotlib.pyplot as plt
from tableone import TableOne
from sksurv.nonparametric import kaplan_meier_estimator

from utils import data_processing, visualization
from utils.simulations import *
from execute import surv_hivae, surv_gan, surv_vae
from sksurv.nonparametric import kaplan_meier_estimator

import os
import json
import uuid
import datetime
import sys
from utils.metrics import fit_cox_model, general_metrics

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)


def adjust_feat_types_for_generator(generator_name, feat_types_dict):
    feat_types_dict_ext = [dict(ft) for ft in feat_types_dict]  # deep copy
    for d in feat_types_dict_ext:
        if d['name'] == "survcens":
            if generator_name == "HI-VAE_weibull" or generator_name == "HI-VAE_weibull_prior":
                d["type"] = 'surv_weibull'
            elif generator_name == "HI-VAE_lognormal":
                d["type"] = 'surv'
            else:
                d["type"] = 'surv_piecewise'
    return feat_types_dict_ext

def setup_unique_working_dir(base_dir="experiments"):
    original_dir = os.getcwd()  # Save original dir
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:8]
    work_dir = os.path.join(base_dir, f"run_{timestamp}_{uid}")
    os.makedirs(work_dir, exist_ok=True)
    return original_dir, work_dir  # Return the original dir

def run(dataset_name):

    generators_dict = {"HI-VAE_weibull" : surv_hivae,
                       "HI-VAE_piecewise" : surv_hivae,
                       "HI-VAE_lognormal" : surv_hivae,
                       "Surv-GAN" : surv_gan,
                       "Surv-VAE" : surv_vae, 
                       "HI-VAE_weibull_prior" : surv_hivae, 
                       "HI-VAE_piecewise_prior" : surv_hivae}

    # Parameters of the optuna study
    metric_optuna = "survival_km_distance" # (or "log_rank_test") metric to optimize in optuna
    epochs = 10000
    n_generated_dataset = 200 # number of generated datasets per fold to compute the metric
    name_config = dataset_name
    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]

    # Set a unique working directory for this job
    original_dir, work_dir = setup_unique_working_dir("parallel_runs")
    best_param_dir = original_dir + "/dataset/" + dataset_name + "/optuna_results"
    best_params_dict = {}
    for generator_name in generators_sel:
        # best_param_file = [item for item in best_param_files if generator_name in item][0]
        for f in os.listdir(best_param_dir):
            if (f.endswith(generator_name + '.json') & (name_config in f)):
                best_param_file = f
        with open(best_param_dir + "/" + best_param_file, "r") as f:
            best_params_dict[generator_name] = json.load(f)

    aug_perc_list = [0, .2, .4, .6]
    synthcity_metrics_sel = ['J-S distance', 'KS test', 'Survival curves distance',
                                'Detection XGB', 'NNDR', 'K-map score']

    # Initialize storage for metrics and results
    synthcity_metrics_res_dict = {generator_name: pd.DataFrame() for generator_name in generators_sel}
    log_p_value_gen_dict = {generator_name: [] for generator_name in generators_sel}
    log_p_value_control_dict = {generator_name: [] for generator_name in generators_sel}
    est_cox_coef_gen_dict = {generator_name: [] for generator_name in generators_sel}
    est_cox_coef_se_gen_dict = {generator_name: [] for generator_name in generators_sel}

    # Initialize result variables
    simu_num = []
    aug_percs = []
    log_p_value_init = []
    est_cox_coef_init = []
    est_cox_coef_se_init = []

    data_file_control= "./dataset/" + dataset_name + "/data_control.csv"
    feat_types_file_control = "./dataset/" + dataset_name + "/data_types_control.csv"
    data_file_treated= "./dataset/" + dataset_name + "/data_treated.csv"
    feat_types_file_treated= "./dataset/" + dataset_name + "/data_types_treated.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    miss_file = "dataset/" + dataset_name + "/Missing.csv"
    true_miss_file = None

    # Load and transform control data
    df_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control,
                                                                                                                feat_types_file_control,
                                                                                                                miss_file, true_miss_file)
    data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
    data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

    # Load and transform treated data
    df_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, miss_file, true_miss_file)
    data_init_treated_encoded = torch.from_numpy(df_init_treated_encoded.values)
    data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)

    fnames = ['time', 'censor'] + pd.read_csv(feat_types_file_control)["name"].to_list()[1:]

    # Format data in dataframe
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)

    # Update the data
    df_init_treated["treatment"] = 1
    df_init_control["treatment"] = 0
    df_init = pd.concat([df_init_control, df_init_treated], ignore_index=True)

    # Set a unique working directory for this job
    original_dir, work_dir = setup_unique_working_dir("parallel_runs")
    os.chdir(work_dir)  # Switch to private work dir
    print("Working directory:", work_dir)
    print("Original directory:", original_dir)

    df_gen_control_dict ={}
    n_samples_control = df_init_control_encoded.shape[0]
    n_samples_control_aug = [int(n_samples_control * (1 + aug_perc)) for aug_perc in aug_perc_list]
    max_n_samples_control_aug = max(n_samples_control_aug)
    data_gen_control_dict = {}
    for generator_name in generators_sel:
        best_params = best_params_dict[generator_name]
        print('\n')
        print("Generator:", generator_name)
        print(best_params)
        print("........ training and generation .........")
        if generator_name in ["HI-VAE_lognormal", "HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            if generator_name in ["HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
                gen_from_prior = True
            else:
                gen_from_prior = False
            feat_types_dict_ext = adjust_feat_types_for_generator(generator_name, feat_types_dict)
            data_gen_control = generators_dict[generator_name].run(df_init_control_encoded, miss_mask_control,
                                                                true_miss_mask_control, feat_types_dict_ext,
                                                                n_generated_dataset, n_generated_sample=max_n_samples_control_aug,
                                                                params=best_params, epochs=epochs, gen_from_prior=gen_from_prior)
        else:
            data_gen_control = generators_dict[generator_name].run(data_init_control, columns=fnames,
                                                                target_column="censor",
                                                                time_to_event_column="time",
                                                                n_generated_dataset=n_generated_dataset,
                                                                n_generated_sample=max_n_samples_control_aug,
                                                                params=best_params)
            
        # data_gen_control est une liste de n_generated_dataset elements de taille max_n_samples_control_aug
        data_gen_control_dict[generator_name] = data_gen_control


    for d in range(len(aug_perc_list)):
        print("Number of samples in generated control group:", n_samples_control_aug[d])
        aug_perc = aug_perc_list[d]
        aug_percs += [aug_perc] * n_generated_dataset

        # Compare the performance of generation in term of p-values between generated control group and intial treated group with different treatment effects
        columns = ['time', 'censor', 'treatment']
        coef_init, _, _, se_init = fit_cox_model(df_init, columns)
        est_cox_coef_init += [coef_init[0]] * n_generated_dataset
        est_cox_coef_se_init += [se_init[0]] * n_generated_dataset

        p_value_init = compute_logrank_test(df_init_control, df_init_treated)
        log_p_value_init += [p_value_init] * n_generated_dataset
        simu_num += [d] * n_generated_dataset

        for generator_name in generators_sel:
            list_df_gen_control = []
            print(generator_name)
            for i in range(n_generated_dataset):
                tmp = data_gen_control_dict[generator_name][i][:n_samples_control_aug[d]]
                if isinstance(tmp, torch.Tensor):
                    df_gen_control = pd.DataFrame(tmp.numpy(), columns=fnames)
                else:
                    df_gen_control = tmp.copy(deep=True)
                df_gen_control["treatment"] = 0
                list_df_gen_control.append(df_gen_control)
            df_gen_control_dict[generator_name] = list_df_gen_control

            log_p_value_gen_list = []
            log_p_value_control_list = []
            est_cox_coef_gen = []
            est_cox_coef_se_gen = []
            # Compare the performance of generation in term of synthcity metric between generated control group and intial control group
            synthcity_metrics_res = general_metrics(df_init_control, list_df_gen_control, generator_name)[synthcity_metrics_sel]
            synthcity_metrics_res_dict[generator_name] = pd.concat([synthcity_metrics_res_dict[generator_name], synthcity_metrics_res])
            for i in range(n_generated_dataset):
                df_gen_control = df_gen_control_dict[generator_name][i]
                log_p_value_gen_list.append(compute_logrank_test(df_gen_control, df_init_treated))
                log_p_value_control_list.append(compute_logrank_test(df_gen_control, df_init_control))

                df_gen = pd.concat([df_gen_control, df_init_treated], ignore_index=True)
                columns = ['time', 'censor', 'treatment']
                coef_gen, _, _, se_gen = fit_cox_model(df_gen, columns)
                est_cox_coef_gen.append(coef_gen[0])
                est_cox_coef_se_gen.append(se_gen[0])

            log_p_value_gen_dict[generator_name] += log_p_value_gen_list
            log_p_value_control_dict[generator_name] += log_p_value_control_list
            est_cox_coef_gen_dict[generator_name] += est_cox_coef_gen
            est_cox_coef_se_gen_dict[generator_name] += est_cox_coef_se_gen


    # SAVE DATAFRAME
    results = pd.DataFrame({'XP_num' : simu_num,
                            "aug_perc" : aug_percs,
                            "log_pvalue_init" : log_p_value_init, 
                            "est_cox_coef_init" : est_cox_coef_init,
                            "est_cox_coef_se_init" : est_cox_coef_se_init})

    for generator_name in generators_sel:
        results["log_pvalue_" + generator_name] = log_p_value_gen_dict[generator_name]
        results[f"log_pvalue_control_{generator_name}"] = log_p_value_control_dict[generator_name]
        results["est_cox_coef_" + generator_name] = est_cox_coef_gen_dict[generator_name]
        results["est_cox_coef_se_" + generator_name] = est_cox_coef_se_gen_dict[generator_name]
        for metric in synthcity_metrics_sel:
            results[metric + "_" + generator_name] = synthcity_metrics_res_dict[generator_name][metric].values

    results.to_csv(f"{original_dir}/dataset/{dataset_name}/results_aug_{metric_optuna}.csv")
   

if __name__ == "__main__":
    dataset_names = ["Aids", "SAS_1", "SAS_2", "SAS_3"]
    dataset_id = int(sys.argv[1])
    run(dataset_names[dataset_id])