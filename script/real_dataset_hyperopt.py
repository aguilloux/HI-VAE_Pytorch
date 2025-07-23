import pandas as pd
import torch

from utils import data_processing
from execute import surv_hivae, surv_gan, surv_vae
import json
import os
import sys
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

import warnings
warnings.filterwarnings("ignore")

def prepare_dataset_dirs(dataset_name):
    base_path = os.path.join("./dataset", dataset_name)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "optuna_results"), exist_ok=True)
    return base_path

def save_parameters(param_path, params):
    with open(param_path, "w") as f:
        for key, value in params.items():
            f.write(f"{key} = {value}\n")

def run_optuna_for_generator(generator_name, data_hi_vae, data, miss_mask, true_miss_mask,
                              feat_types_dict, n_generated_dataset, n_splits, n_trials, columns, epochs, study_path):
    generator_func = {
        "HI-VAE_weibull": surv_hivae,
        "HI-VAE_piecewise": surv_hivae,
        "HI-VAE_lognormal": surv_hivae,
        "Surv-GAN": surv_gan,
        "Surv-VAE": surv_vae
    }[generator_name]

    feat_types_dict_ext = [dict(ft) for ft in feat_types_dict]  # deep copy
    for d in feat_types_dict_ext:
        if d['name'] == "survcens":
            if generator_name == "HI-VAE_weibull":
                d["type"] = 'surv_weibull'
            elif generator_name == "HI-VAE_lognormal":
                d["type"] = 'surv'
            else:
                d["type"] = 'surv_piecewise'

    if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_lognormal"]:
        best_params, _ = generator_func.optuna_hyperparameter_search(data_hi_vae, miss_mask, true_miss_mask, feat_types_dict_ext,
                                                                     n_generated_dataset, n_splits=n_splits, n_trials=n_trials,
                                                                     columns=columns, generator_name=generator_name, epochs=epochs,
                                                                     study_name=study_path)
    else:
        best_params, _ = generator_func.optuna_hyperparameter_search(data, columns=columns, target_column="censor",
                                                                     time_to_event_column="time",
                                                                     n_generated_dataset=n_generated_dataset,
                                                                     n_splits=n_splits, n_trials=n_trials,
                                                                     study_name=study_path)

    return best_params

def kaplan_meier_estimation(surv_data, label=None, ax=None):
    """Plot Kaplan-Meier curve with confidence interval."""
    surv_time = surv_data['time'].values
    surv_event = surv_data['censor'].values.astype(bool)
    uniq_time, surv_prob, conf_int = kaplan_meier_estimator(surv_event, surv_time, conf_type="log-log")

    ax.step(uniq_time, surv_prob, where="post", label=label)
    ax.fill_between(uniq_time, conf_int[0], conf_int[1], alpha=0.25, step="post")

def verify_hyperopt(generators, data_hi_vae, data, miss_mask, true_miss_mask,feat_types_dict, n_generated_dataset, columns, res_file, best_params_dict, epochs):
    data_gen_control_dict_best_params = {}
    data_gen_control_dict = {}
    for generator_name in generators:
        print("=" * 100)
        generator_func = {
            "HI-VAE_weibull": surv_hivae,
            "HI-VAE_piecewise": surv_hivae,
            "HI-VAE_lognormal": surv_hivae,
            "Surv-GAN": surv_gan,
            "Surv-VAE": surv_vae
        }[generator_name]
        best_params = best_params_dict[generator_name]
        if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise"]:
            feat_types_dict_ext = adjust_feat_types_for_generator(generator_name, feat_types_dict)

            print("Generate data by {} with best params".format(generator_name))
            data_gen_control_dict_best_params[generator_name] = generator_func.run(data_hi_vae, miss_mask,
                                                                                   true_miss_mask, feat_types_dict_ext,
                                                                                   n_generated_dataset, params=best_params,
                                                                                   epochs=epochs)
            print("Generate data by {} with defaut params".format(generator_name))
            data_gen_control_dict[generator_name] = generator_func.run(data_hi_vae, miss_mask,
                                                                       true_miss_mask, feat_types_dict_ext, n_generated_dataset,
                                                                       epochs=epochs)
        else:
            print("Generate data by {} with best params".format(generator_name))
            data_gen_control_dict_best_params[generator_name] = generator_func.run(data, columns=columns,
                                                                                   target_column="censor", time_to_event_column="time",
                                                                                   n_generated_dataset=n_generated_dataset,
                                                                                   params=best_params)
            print("Generate data by {} with defaut params".format(generator_name))
            data_gen_control_dict[generator_name] = generator_func.run(data, columns=columns,
                                                                       target_column="censor", time_to_event_column="time",
                                                                       n_generated_dataset=n_generated_dataset)

    # COMPARE THE RESULTS BETWEEN THE BEST PARAMS WITH DEFAULT ONES
    _, axs = plt.subplots(1, 2, figsize=(20, 5))
    sel_dataset_idx = 0
    for i, generator_name in enumerate(generators):
        df_syn_sel = pd.DataFrame(data_gen_control_dict[generator_name][sel_dataset_idx].numpy(), columns=columns)
        kaplan_meier_estimation(df_syn_sel, label="Generated Control Group " + generator_name, ax=axs[0])
        df_syn_sel_best_params = pd.DataFrame(data_gen_control_dict_best_params[generator_name][sel_dataset_idx].numpy(), columns=columns)
        kaplan_meier_estimation(df_syn_sel_best_params, label="Generated Control Group " + generator_name, ax=axs[1])

    control = pd.DataFrame(data.numpy(), columns=columns)
    for ax, title in zip(axs,
                         ["Survival Curves with Confidence Intervals with default setting",
                          "Survival Curves with Confidence Intervals with best params"]):
        kaplan_meier_estimation(control, label="Initial Control Group", ax=ax)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=15)
        ax.set_xlabel("Time", fontweight="semibold")
        ax.set_ylabel("Survival Probability", fontweight="semibold")
        ax.set_title(title, fontweight="bold")

    plt.savefig(res_file)

def adjust_feat_types_for_generator(generator_name, feat_types_dict):
    feat_types_dict_ext = [dict(ft) for ft in feat_types_dict]  # deep copy
    for d in feat_types_dict_ext:
        if d['name'] == "survcens":
            if generator_name == "HI-VAE_weibull":
                d["type"] = 'surv_weibull'
            elif generator_name == "HI-VAE_lognormal":
                d["type"] = 'surv'
            else:
                d["type"] = 'surv_piecewise'
    return feat_types_dict_ext

def run(dataset_name):

    base_path = prepare_dataset_dirs(dataset_name)

     ## DATA LOADING
    data_file_control= "./dataset/" + dataset_name + "/data_control.csv"
    feat_types_file_control = "./dataset/" + dataset_name + "/data_types_control.csv"
    data_file_treated= "./dataset/" + dataset_name + "/data_treated.csv"
    feat_types_file_treated= "./dataset/" + dataset_name + "/data_types_treated.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    m_perc = 10
    mask = 1
    miss_file = "dataset/" + dataset_name + "/Missing{}_{}.csv".format(m_perc, mask)
    true_miss_file = None

    fnames = ['time', 'censor'] + pd.read_csv(feat_types_file_control)["name"].to_list()[1:]
    # Load and transform control data
    df_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control, feat_types_file_control, miss_file, true_miss_file)
    data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
    data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

    # Load and transform treated data
    df_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, miss_file, true_miss_file)
    data_init_treated_encoded = torch.from_numpy(df_init_treated_encoded.values)
    data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)

    # Format data in dataframe
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)

    # Update dataframe
    df_init_treated["treatment"] = 1
    df_init_control["treatment"] = 0

    ## HYPER-PARAMETER OPTIMIZATION
    # Parameters of the optuna study
    metric_optuna = "survival_km_distance" # (or "log_rank_test") metric to optimize in optuna
    n_splits = 5 # number of splits for cross-validation
    n_trials = 150
    epochs = 10000
    n_generated_dataset = 200 # number of generated datasets per fold to compute the metric
    name_config = dataset_name
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE"]

    # HYPER-PARAMETER OPTIMIZATION
    best_params_dict = {}
    for generator_name in generators_sel:
        study_path = os.path.join(base_path, "optuna_results", "optuna_study_{}_ntrials{}_{}_{}".format(name_config, n_trials, metric_optuna, generator_name))
        best_params_file = os.path.join(base_path, "optuna_results", "best_params_{}_ntrials{}_{}_{}.json".format(name_config, n_trials, metric_optuna, generator_name))
        db_file = study_path + ".db"
        if os.path.exists(db_file):
            print("This optuna study ({}) already exists for {}. We will use this existing file.".format(db_file, generator_name))
            with open(best_params_file, "r") as f:
                best_params_dict[generator_name] = json.load(f)
        else:
            best_params = run_optuna_for_generator(generator_name, df_init_control_encoded, data_init_control, miss_mask_control,
                                                    true_miss_mask_control, feat_types_dict, n_generated_dataset,
                                                    n_splits, n_trials, fnames, epochs, study_path)
            best_params_dict[generator_name] = best_params
            with open(best_params_file, "w") as f:
                json.dump(best_params, f)

    # COMPARE THE RESULTS BETWEEN THE BEST PARAMS WITH DEFAULT ONES
    res_file = "./dataset/" + dataset_name + "/hyperopt_independent.jpeg"
    verify_hyperopt(generators = generators_sel,
                    data_hi_vae = df_init_control_encoded,
                    data = data_init_control,
                    miss_mask = miss_mask_control,
                    true_miss_mask = true_miss_mask_control,
                    feat_types_dict = feat_types_dict,
                    n_generated_dataset = n_generated_dataset,
                    columns = fnames,
                    res_file = res_file,
                    best_params_dict = best_params_dict,
                    epochs=epochs)

if __name__ == "__main__":
    # for dataset_name in ["gbsb2", "Aids", "SAS_1", "SAS_2", "SAS_3"]:
    for dataset_name in ["gbsb2"]:
        run(dataset_name)