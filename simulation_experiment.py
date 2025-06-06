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
from utils.metrics import fit_cox_model, general_metrics

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)

def true_univ_coef(treatment_effect, independent = True, feature_types_list = ["pos", "real", "cat"],
                   n_features_bytype = 4, n_active_features = 3 , p_treated = 0.5, shape_T = 2, shape_C = 2,
                   scale_C = 6., scale_C_indep = 2.5, data_types_create = True, seed=0):

    # Compute univariate treatment effects
    n_samples = 100000
    seed = int(np.random.randint(1000, size = 1))
    control, treated, types = simulation(treatment_effect, n_samples,
                                         independent = independent, n_features_bytype  = n_features_bytype,
                                         n_active_features = n_active_features,
                                         feature_types_list = feature_types_list,
                                         shape_T = shape_T , shape_C = shape_C,
                                         scale_C = scale_C , scale_C_indep = scale_C_indep, seed=seed)

    df_init = pd.concat([control, treated], ignore_index=True)
    columns = ['time', 'censor', 'treatment']
    coef_init = fit_cox_model(df_init, columns)[0]

    return coef_init[0]

def run():
    # Simulate the initial data
    n_samples = 600
    n_features_bytype = 4
    n_active_features = 3 
    treatment_effect = 0.
    p_treated = 0.5
    shape_T = 2.
    shape_C = 2.
    scale_C = 6.
    scale_C_indep = 4.5
    feature_types_list = ["pos", "real", "cat"]
    independent = False
    data_types_create = True

    control, treated, types = simulation(treatment_effect, n_samples, independent, feature_types_list,
                                         n_features_bytype, n_active_features, p_treated, shape_T,
                                         shape_C, scale_C, scale_C_indep, data_types_create, seed=0)

    control = control.drop(columns='treatment')
    treated = treated.drop(columns='treatment')

    if not os.path.exists("./dataset"):
        os.makedirs("./dataset/")

    # Save the data
    dataset_name = "Simulations"
    if not os.path.exists("./dataset/" + dataset_name):
        os.makedirs("./dataset/" + dataset_name)
    data_file_control= "./dataset/" + dataset_name + "/data_control.csv"
    feat_types_file_control = "./dataset/" + dataset_name + "/data_types_control.csv"
    data_file_treated= "./dataset/" + dataset_name + "/data_treated.csv"
    feat_types_file_treated= "./dataset/" + dataset_name + "/data_types_treated.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    miss_file = "dataset/" + dataset_name + "/Missing.csv"
    true_miss_file = None

    control.to_csv(data_file_control,index=False , header=False)
    types.to_csv(feat_types_file_control)
    treated.to_csv(data_file_treated,index=False , header=False)
    types.to_csv(feat_types_file_treated)


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

    fnames = types['name'][:-1].tolist()
    fnames.append("time")#.append("censor")
    fnames.append("censor")


    # Format data in dataframe
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)

    # Update the data
    df_init_treated["treatment"] = 1
    df_init_control["treatment"] = 0
    df_init = pd.concat([df_init_control, df_init_treated], ignore_index=True)

    # Parameters of the optuna study
    multiplier_trial = 10 # multiplier for the number of trials
    n_splits = 5 # number of splits for cross-validation
    n_generated_dataset = 1 # number of generated datasets per fold to compute the metric
    name_config = "simu_N{}_nfeat{}_t{}".format(n_samples, n_features_bytype, int(treatment_effect*10))

    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE"]
    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]
    # generators_sel = ["Surv-GAN", "Surv-VAE"]
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
            print("This optuna study ({}) already exists for {}. We will use this existing file.".format(db_file, generator_name))
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
                best_params, study = generators_dict[generator_name].optuna_hyperparameter_search(df_init_control_encoded,
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

    # RUN WITH DEFAULT PARAMETERS
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
            data_gen_control_dict[generator_name] = generators_dict[generator_name].run(df_init_control_encoded, miss_mask_control, true_miss_mask_control, feat_types_dict_ext, n_generated_dataset)
        else:
            data_gen_control_dict[generator_name] = generators_dict[generator_name].run(data_init_control, columns=fnames, target_column="censor", time_to_event_column="time", n_generated_dataset=n_generated_dataset)

    # RUN WITH BEST PARAMETERS
    best_params_dict = {}
    for generator_name in generators_sel:
        n_trials = int(multiplier_trial * generators_dict[generator_name].get_n_hyperparameters(generator_name))
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
            data_gen_control_dict_best_params[generator_name] = generators_dict[generator_name].run(df_init_control_encoded,
                                                                                                    miss_mask_control,
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
    kaplan_meier_estimation(control, label="Initial Control Group", ax=axs[0])
    kaplan_meier_estimation(treated, label="Treatment Group", ax=axs[0])
    p_value_default_str = "-Log rank pvalues: \n Initial {:.2f}"
    p_value_default_value = [compute_logrank_test(control, treated)]

    sel_dataset_idx = 0
    for i, generator_name in enumerate(generators_sel):
        df_syn_sel = pd.DataFrame(data_gen_control_dict[generator_name][sel_dataset_idx].numpy(), columns=fnames)
        kaplan_meier_estimation(df_syn_sel, label="Generated Control Group " + generator_name, ax=axs[0])
        p_value_default_str += "\n " + generator_name + " {:.2f}"
        p_value_default_value.append(compute_logrank_test(df_syn_sel, treated))

    axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize=15)
    axs[0].set_xlabel("Time", fontweight="semibold")
    axs[0].set_ylabel("Survival Probability", fontweight="semibold")
    axs[0].set_title("Survival Curves with Confidence Intervals with default setting", fontweight="bold")

    kaplan_meier_estimation(control, label="Initial Control Group", ax=axs[1])
    kaplan_meier_estimation(treated, label="Treatment Group", ax=axs[1])
    p_value_best_str = "-Log rank pvalues: \n Initial {:.2f}"
    p_value_best_value = [compute_logrank_test(control, treated)]

    sel_dataset_idx = 0
    for i, generator_name in enumerate(generators_sel):
        df_syn_sel = pd.DataFrame(data_gen_control_dict_best_params[generator_name][sel_dataset_idx].numpy(), columns=fnames)
        kaplan_meier_estimation(df_syn_sel, label="Generated Control Group " + generator_name, ax=axs[1])
        p_value_best_str += "\n " + generator_name + " {:.2f}"
        p_value_best_value.append(compute_logrank_test(df_syn_sel, treated))

    axs[1].set_ylim(0, 1)
    axs[1].legend(fontsize=15)
    axs[1].set_xlabel("Time", fontweight="semibold")
    axs[1].set_ylabel("Survival Probability", fontweight="semibold")
    axs[1].set_title("Survival Curves with Confidence Intervals with best params", fontweight="bold")
    plt.savefig("./dataset/" + dataset_name + "/hyperopt_independent_n_samples_" + str(n_samples) + "n_features_bytype_" + str(n_features_bytype) + ".jpeg")

    # MONTE-CARLO EXPERIMENT
    treat_effects = np.arange(0., 1.1, 0.2)
    n_generated_dataset = 50
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE"]
    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]
    synthcity_metrics_sel = ['J-S distance', 'KS test', 'Survival curves distance', 'Detection XGB', 'NNDR', 'K-map score']
    n_MC_exp = 100

    simu_num = []
    D_control = []
    D_treated = []
    coef_init_univ_list = []
    H0_coef = []
    log_p_value_init = []
    log_p_value_gen_dict = {}
    est_cox_coef_init = []
    est_cox_coef_se_init = []
    est_cox_coef_gen_dict = {}
    est_cox_coef_se_gen_dict = {}
    synthcity_metrics_res_dict = {}
    for generator_name in generators_sel:
        log_p_value_gen_dict[generator_name] = []
        est_cox_coef_gen_dict[generator_name] = []
        est_cox_coef_se_gen_dict[generator_name] = []
        synthcity_metrics_res_dict[generator_name] = pd.DataFrame()

    seed = 0
    for t in np.arange(len(treat_effects)):
        treatment_effect = treat_effects[t]
        coef_init_univ = true_univ_coef(treatment_effect, independent, feature_types_list,
                                         n_features_bytype, n_active_features, p_treated, shape_T,
                                         shape_C, scale_C, scale_C_indep, data_types_create, seed=seed)
        print("Treatment_effect", treatment_effect)
        for m in np.arange(n_MC_exp):
            if m % 10 == 0:
                print("Monte-Carlo experiment", m)
            # To make sure the difference between simulated dataset, increase seed value each time
            seed += 1
            control, treated, types = simulation(treatment_effect, n_samples, independent, feature_types_list,
                                                 n_features_bytype, n_active_features, p_treated, shape_T, shape_C,
                                                 scale_C, scale_C_indep, data_types_create, seed=seed)


            control = control.drop(columns='treatment')
            treated = treated.drop(columns='treatment')
            
            data_file_control = "./dataset/" + dataset_name + "/data_control.csv"
            data_file_treated = "./dataset/" + dataset_name + "/data_treated.csv"
            feat_types_file_control = "./dataset/" + dataset_name + "/data_types_control.csv"
            feat_types_file_treated= "./dataset/" + dataset_name + "/data_types_treated.csv"
        
            control.to_csv(data_file_control, index=False , header=False)
            treated.to_csv(data_file_treated, index=False , header=False)
            types.to_csv(feat_types_file_control)
            types.to_csv(feat_types_file_treated)


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

            # Format data in dataframe
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
            columns = ['time', 'censor', 'treatment']
            coef_init, _, _, se_init = fit_cox_model(df_init, columns)

            est_cox_coef_init += [coef_init[0]] * n_generated_dataset
            est_cox_coef_se_init += [se_init[0]] * n_generated_dataset

            p_value_init = compute_logrank_test(df_init_control, df_init_treated)
            log_p_value_init += [p_value_init] * n_generated_dataset
            H0_coef += [treatment_effect] * n_generated_dataset
            simu_num += [t * n_MC_exp + m] * n_generated_dataset
            D_control += [control['censor'].sum()] * n_generated_dataset
            D_treated += [treated['censor'].sum()] * n_generated_dataset
            coef_init_univ_list += [coef_init_univ] * n_generated_dataset

            for generator_name in generators_sel:
                best_params = best_params_dict[generator_name]
                if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise"]:
                    feat_types_dict_ext = feat_types_dict.copy()
                    for i in range(len(feat_types_dict)):
                        if feat_types_dict_ext[i]['name'] == "survcens":
                            if generator_name in["HI-VAE_weibull"]:
                                feat_types_dict_ext[i]["type"] = 'surv_weibull'
                            else:
                                feat_types_dict_ext[i]["type"] = 'surv_piecewise'
                    data_gen_control = generators_dict[generator_name].run(df_init_control_encoded, miss_mask_control, true_miss_mask_control,
                                                                        feat_types_dict_ext, n_generated_dataset, 
                                                                        params=best_params)
                else:
                    data_gen_control = generators_dict[generator_name].run(data_init_control, columns=fnames, 
                                                                        target_column="censor", time_to_event_column="time", 
                                                                        n_generated_dataset=n_generated_dataset, params=best_params)

                log_p_value_gen_list = []
                est_cox_coef_gen = []
                est_cox_coef_se_gen = []
                list_df_gen_control = []

                for i in range(n_generated_dataset):
                    df_gen_control = pd.DataFrame(data_gen_control[i].numpy(), columns=fnames)
                    df_gen_control["treatment"] = 0
                    list_df_gen_control.append(df_gen_control)
                    log_p_value_gen_list.append(compute_logrank_test(df_gen_control, treated))
                    df_gen = pd.concat([df_gen_control, df_init_treated], ignore_index=True)
                    columns = ['time', 'censor', 'treatment']
                    coef_gen, _, _, se_gen = fit_cox_model(df_gen, columns)
                    est_cox_coef_gen.append(coef_gen[0])
                    est_cox_coef_se_gen.append(se_gen[0])

                synthcity_metrics_res = general_metrics(df_init_control, list_df_gen_control, generator_name)[synthcity_metrics_sel]

                log_p_value_gen_dict[generator_name] += log_p_value_gen_list
                est_cox_coef_gen_dict[generator_name] += est_cox_coef_gen
                est_cox_coef_se_gen_dict[generator_name] += est_cox_coef_se_gen
                synthcity_metrics_res_dict[generator_name] = pd.concat([synthcity_metrics_res_dict[generator_name], 
                                                                        synthcity_metrics_res])

    # SAVE DATAFRAME
    results = pd.DataFrame({'XP_num' : simu_num,
                            "D_control" : D_control,
                            "D_treated" : D_treated,
                            "H0_coef_univ" : coef_init_univ_list,
                            "H0_coef" : H0_coef,
                            "log_pvalue_init" : log_p_value_init, 
                            "est_cox_coef_init" : est_cox_coef_init,
                            "est_cox_coef_se_init" : est_cox_coef_se_init})

    for generator_name in generators_sel:
        results["log_pvalue_" + generator_name] = log_p_value_gen_dict[generator_name]
        results["est_cox_coef_" + generator_name] = est_cox_coef_gen_dict[generator_name]
        results["est_cox_coef_se_" + generator_name] = est_cox_coef_se_gen_dict[generator_name]
        for metric in synthcity_metrics_sel:
            results[metric + "_" + generator_name] = synthcity_metrics_res_dict[generator_name][metric].values

    results.to_csv("./dataset/" + dataset_name + "/results_n_samples_" + str(n_samples) + "n_features_bytype_" + str(n_features_bytype) + ".csv")


if __name__ == "__main__":
    run()