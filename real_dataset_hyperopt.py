import pandas as pd
import torch

from utils import data_processing
from execute import surv_hivae, surv_gan, surv_vae
import json
import os
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

import warnings
warnings.filterwarnings("ignore")

def run(dataset_name):

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
    data_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control, feat_types_file_control, miss_file, true_miss_file)
    data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

    # Load and transform treated data
    data_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, miss_file, true_miss_file)
    data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)

    # Format data in dataframe
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)

    # Update dataframe
    df_init_treated["treatment"] = 1
    df_init_control["treatment"] = 0

    ## HYPER-PARAMETER OPTIMIZATION
    # Parameters of the optuna study
    name_config = dataset_name
    multiplier_trial = 10 # multiplier for the number of trials
    n_splits = 5 # number of splits for cross-validation
    n_generated_dataset = 1 # number of generated datasets per fold to compute the metric

    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE"]
    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]
    generators_dict = {"HI-VAE_weibull" : surv_hivae,
                    "HI-VAE_piecewise" : surv_hivae,
                    "Surv-GAN" : surv_gan,
                    "Surv-VAE" : surv_vae}


    best_params_dict, study_dict = {}, {}
    for generator_name in generators_sel:
        n_trials = int(multiplier_trial * generators_dict[generator_name].get_n_hyperparameters(generator_name))
        print("{} trials for {}...".format(n_trials, generator_name))
        db_file = "optuna_results/optuna_study_{}_ntrials{}_{}.db".format(name_config, n_trials, generator_name)
        if os.path.exists(db_file):
            print("This optuna study ({}) already exists for {}. Please change the name of the study or remove the file to create a new one.".format(db_file, generator_name))
        else: 
            print("Creating new optuna study for {}...".format(generator_name))
            if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise"]:
                feat_types_dict_ext = feat_types_dict.copy()
                for i in range(len(feat_types_dict)):
                    if feat_types_dict_ext[i]['name'] == "survcens":
                        if generator_name in["HI-VAE_weibull"]:
                            feat_types_dict_ext[i]["type"] = 'surv_weibull'
                        else:
                            feat_types_dict_ext[i]["type"] = 'surv_piecewise'
                best_params, study = generators_dict[generator_name].optuna_hyperparameter_search(data_init_control_encoded, 
                                                                                                data_init_control,
                                                                                                miss_mask_control, 
                                                                                                true_miss_mask_control,
                                                                                                feat_types_dict_ext, 
                                                                                                n_generated_dataset, 
                                                                                                n_splits=n_splits,
                                                                                                n_trials=n_trials, 
                                                                                                columns=fnames,
                                                                                                generator_name=generator_name,
                                                                                                epochs=1000,
                                                                                                study_name="optuna_results/optuna_study_{}_ntrials{}_{}".format(name_config, n_trials, generator_name))
                best_params_dict[generator_name] = best_params
                study_dict[generator_name] = study
                with open("optuna_results/best_params_{}_ntrials{}_{}.json".format(name_config, n_trials, generator_name), "w") as f:
                    json.dump(best_params, f)
            else: 
                best_params, study = generators_dict[generator_name].optuna_hyperparameter_search(data_init_control, 
                                                                                                columns=fnames, 
                                                                                                target_column="censor", 
                                                                                                time_to_event_column="time", 
                                                                                                n_generated_dataset=n_generated_dataset, 
                                                                                                n_splits=n_splits,
                                                                                                n_trials=n_trials, 
                                                                                                study_name="optuna_results/optuna_study_{}_ntrials{}_{}".format(name_config, n_trials, generator_name))
                best_params_dict[generator_name] = best_params
                study_dict[generator_name] = study
                with open("optuna_results/best_params_{}_ntrials{}_{}.json".format(name_config, n_trials, generator_name), "w") as f:
                    json.dump(best_params, f)

    ## RUN WITH DEFAULT PARAMETERS
    # the datasets used for training is data_init_control
    n_generated_dataset = 50
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE"]
    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]
    data_gen_control_dict = {}
    for generator_name in generators_sel:
        print("=" * 100)
        print("Generate data by " + generator_name)
        if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise"]:
            feat_types_dict_ext = feat_types_dict.copy()
            for i in range(len(feat_types_dict)):
                if feat_types_dict_ext[i]['name'] == "survcens":
                    if generator_name in["HI-VAE_weibull"]:
                        feat_types_dict_ext[i]["type"] = 'surv_weibull'
                    else:
                        feat_types_dict_ext[i]["type"] = 'surv_piecewise'
            data_gen_control_dict[generator_name] = generators_dict[generator_name].run(data_init_control_encoded, data_init_control, fnames, miss_mask_control, true_miss_mask_control, feat_types_dict_ext, n_generated_dataset)
        else:
            data_gen_control_dict[generator_name] = generators_dict[generator_name].run(data_init_control, columns=fnames, target_column="censor", time_to_event_column="time", n_generated_dataset=n_generated_dataset)

    ## RUN WITH BEST PARAMETERS
    best_params_dict = {}
    for generator_name in generators_sel:
        with open("optuna_results/best_params_{}_ntrials{}_{}.json".format(name_config, n_trials, generator_name), "r") as f:
            best_params_dict[generator_name] = json.load(f)

    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE"]
    data_gen_control_dict_best_params = {}
    for generator_name in generators_sel:
        print("=" * 100)
        print("Generate data by " + generator_name)
        best_params = best_params_dict[generator_name]
        if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise"]:
            feat_types_dict_ext = feat_types_dict.copy()
            for i in range(len(feat_types_dict)):
                if feat_types_dict_ext[i]['name'] == "survcens":
                    if generator_name in["HI-VAE_weibull"]:
                        feat_types_dict_ext[i]["type"] = 'surv_weibull'
                    else:
                        feat_types_dict_ext[i]["type"] = 'surv_piecewise'
            data_gen_control_dict_best_params[generator_name] = generators_dict[generator_name].run(data_init_control_encoded, 
                                                                                                    data_init_control, 
                                                                                                    fnames, miss_mask_control, 
                                                                                                    true_miss_mask_control, 
                                                                                                    feat_types_dict_ext, 
                                                                                                    n_generated_dataset, 
                                                                                                    params=best_params)
        else:
            data_gen_control_dict_best_params[generator_name] = generators_dict[generator_name].run(data_init_control, 
                                                                                                    columns=fnames, 
                                                                                                    target_column="censor", 
                                                                                                    time_to_event_column="time", 
                                                                                                    n_generated_dataset=n_generated_dataset, 
                                                                                                    params=best_params)
            
    # COMPARE THE RESULTS BETWEEN THE BEST PARAMS WITH DEFAULT ONES
    def kaplan_meier_estimation(surv_data, label=None, ax=None):
        surv_time  = surv_data['time'].values
        surv_ind = surv_data['censor'].values.astype(bool)
        uniq_time, surv_prob, conf_int = kaplan_meier_estimator(surv_ind, surv_time, conf_type="log-log")

        ax.step(uniq_time, surv_prob, where="post", label=label)
        ax.fill_between(uniq_time, conf_int[0], conf_int[1], alpha=0.25, step="post")

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    kaplan_meier_estimation(df_init_control, label="Initial Control Group", ax=axs[0])
    kaplan_meier_estimation(df_init_treated, label="Treatment Group", ax=axs[0])

    axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize=15)
    axs[0].set_xlabel("Time", fontweight="semibold")
    axs[0].set_ylabel("Survival Probability", fontweight="semibold")
    axs[0].set_title("Survival Curves with Confidence Intervals with default setting", fontweight="bold")

    kaplan_meier_estimation(df_init_control, label="Initial Control Group", ax=axs[1])
    kaplan_meier_estimation(df_init_treated, label="Treatment Group", ax=axs[1])

    sel_dataset_idx = 0
    for i, generator_name in enumerate(generators_sel):
        df_syn_sel = pd.DataFrame(data_gen_control_dict_best_params[generator_name][sel_dataset_idx].numpy(), columns=fnames)
        kaplan_meier_estimation(df_syn_sel, label="Generated Control Group " + generator_name, ax=axs[1])

    axs[1].set_ylim(0, 1)
    axs[1].legend(fontsize=15)
    axs[1].set_xlabel("Time", fontweight="semibold")
    axs[1].set_ylabel("Survival Probability", fontweight="semibold")
    axs[1].set_title("Survival Curves with Confidence Intervals with best params", fontweight="bold")
    plt.savefig("./dataset/" + dataset_name + "/hyperopt.jpeg")


if __name__ == "__main__":
    # for dataset_name in ["Aids", "SAS_1", "SAS_2", "SAS_3"]:
    for dataset_name in ["SAS_2", "SAS_3"]:
        run(dataset_name)