import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")
from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)

from utils.simulations import *
from utils.metrics import fit_cox_model, estimate_agreement, decision_agreement, standardized_difference, ci_overlap
from utils.visualization import visualize_replicability_perf

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

    # MONTE-CARLO EXPERIMENT
    treat_effects = np.arange(0., 1.1, 0.8)
    n_MC_exp = 3

    simu_num = []
    D_control = []
    D_treated = []
    coef_init_univ_list = []
    XP_num = []

    # Save the data
    dataset_name = "Simulations"

    seed = 0
    for t in np.arange(len(treat_effects)):
        treatment_effect = treat_effects[t]
        coef_init_univ = true_univ_coef(treatment_effect, n_features_bytype = n_features_bytype,
                                        n_active_features = n_active_features, feature_types_list = ["pos", "real", "cat"],
                                        shape_T = 2, shape_C = 2, scale_C = 6., scale_C_indep = 2.5, seed=seed)
        for m in np.arange(n_MC_exp):
            # To make sure the difference between simulated dataset, increase seed value each time
            seed += 1
            control, treated, types = simulation(treatment_effect, n_samples, independent = False, feature_types_list = ["pos", "real", "cat"],
                                                    n_features_bytype = 4, n_active_features = 3 , p_treated = 0.5, shape_T = 2, shape_C = 2,
                                                    scale_C = 6., scale_C_indep = 2.5, data_types_create = True, seed=seed)
            
            control = control.drop(columns='treatment')
            treated = treated.drop(columns='treatment')
            D_control.append(control['censor'].sum())
            D_treated.append(treated['censor'].sum()) 


            simu_num.append(t * n_MC_exp + m)
            XP_num += [t * n_MC_exp + m] * 50

        coef_init_univ_list += [coef_init_univ] * n_MC_exp

    # SAVE DATAFRAME
    results_true_coef = pd.DataFrame({'XP_num' : simu_num,
                            "D_control" : D_control,
                            "D_treated" : D_treated,
                            "H0_coef_univ" : coef_init_univ_list})

    results = pd.read_csv("./dataset/" + dataset_name + "/results_independent_n_samples_" + str(n_samples) + "n_features_bytype_" + str(n_features_bytype) + ".csv")
    results["XP_num"] = XP_num
    results = pd.merge(results, results_true_coef, on="XP_num")

    #### COMPARE WITH TRUE COEFS
    results_ext = results.copy(deep=True)
    results_ext["reject_H0_init"] = results_ext['log_pvalue_init'] > -np.log(0.05)
    power_init = results_ext.groupby("H0_coef").mean()["reject_H0_init"].to_numpy()
    plt.plot(treat_effects, power_init, '-', label = "Init.")

    alpha = 0.05
    expected_power = []
    for treat_effect in treat_effects:
        tmp = results[results.H0_coef == treat_effect].mean()
        Dc = tmp["D_control"]
        Dt = tmp["D_treated"]
        coef_init_univ = tmp["H0_coef_univ"]
        power = cpower(Dc , Dt, coef_init_univ, alpha)
        expected_power.append(power)
    plt.scatter(treat_effects, expected_power, label = "Expectation")

    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE"]
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]
    for generator_name in generators_sel:
        results_ext["reject_H0_gen_" + generator_name] = results_ext['log_pvalue_' + generator_name] > -np.log(0.05)
        power_gen = results_ext.groupby("H0_coef").mean()["reject_H0_gen_" + generator_name].to_numpy()
        plt.plot(treat_effects, power_gen, '--', marker='o',label = generator_name)

    plt.xlabel("Treatment effect")
    plt.ylabel("Level/power")
    plt.legend()
    plt.savefig("./dataset/" + dataset_name + "/results_independent_n_samples_" + str(n_samples) + "n_features_bytype_" + str(n_features_bytype) + ".jpeg")

    #### SYNTHCITY METRICS 
    metrics=[['J-S distance', "min"], ['KS test', "max"]]
    num_metrics = len(metrics)
    n_learners = len(generators_sel)
    fig, axs = plt.subplots(1, num_metrics, figsize=(3 * num_metrics * n_learners, 6))

    if num_metrics == 1:
        axs = [axs]  # ensure axs is iterable

    for i, ax in enumerate(axs):
        # Format axis spines
        metric_name, opt = metrics[i]
        metric_df = pd.DataFrame()
        for generator_name in generators_sel:
            metric_df = pd.concat([metric_df, pd.DataFrame(np.array([[generator_name] * results.shape[0], 
                                                            results[metric_name + "_" + generator_name]]).T,
                                                            columns=['generator', metric_name])])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        sns.boxplot(data=metric_df, x='generator', y=metric_name, ax=ax,
                    linewidth = 3, saturation = 1, palette = 'colorblind', 
                    width = 1, gap = 0.15, whis = 0.8, linecolor="Black")
        ax.set_xlabel('')
        ax.set_ylabel(metric_name, fontsize=20, fontweight="semibold")
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        if opt == "max":
            ax.legend(title='Maximize \u2191', title_fontsize=15)
        else:
            ax.legend(title='Minimize \u2193', title_fontsize=15)
    plt.tight_layout(pad=3)
    plt.savefig("./dataset/" + dataset_name + "/results_synthetic_metrics_independent_n_samples_" + str(n_samples) + "n_features_bytype_" + str(n_features_bytype) + ".jpeg")

    #### REPLICABILITY 
    score_df = pd.DataFrame(columns=["Generator", "Nb generated datasets", "Estimate agreement", "Decision agreement", "Standardized difference", "CI overlap"])
    for treatment_effect in treat_effects:
        results_treat = results[results.H0_coef == treatment_effect]
        cox_init = results_treat[["est_cox_coef_init", "est_cox_coef_se_init"]].drop_duplicates().values
        for m in range(n_MC_exp):
            results_MC = results_treat[(results_treat[["est_cox_coef_init", "est_cox_coef_se_init"]] == cox_init[m]).all(axis=1)]
            coef_init, se_init = results_MC[["est_cox_coef_init", "est_cox_coef_se_init"]].drop_duplicates().values[0]
            ci_init = (coef_init - 1.96 * se_init, coef_init + 1.96 * se_init)

            for generator in generators_sel:
                coef_syn, se_syn = results_MC[["est_cox_coef_" + generator, "est_cox_coef_se_" + generator]].values.T
                max_len_samples = len(coef_syn)
                list_len_samples = np.arange(int(.2 * max_len_samples), max_len_samples, int(.2 * max_len_samples)).tolist()
                if max_len_samples not in list_len_samples:
                    list_len_samples += [max_len_samples]
                for j in list_len_samples:
                    coef_syn_, se_syn_ = np.array(coef_syn)[:j], np.array(se_syn)[:j]
                    coef_syn_mean = coef_syn_.mean()
                    var_syn_mean = (se_syn_**2).mean()
                    # imputation_var_syn = (1 / (len(coef_syn) - 1)) * np.sum([(coef_syn_ - coef_syn_mean)**2 for coef_syn_ in coef_syn])
                    # adjusted_var_syn = (imputation_var_syn / len(coef_syn)) + var_syn_mean
                    adjusted_var_syn = (1/j + 1) * var_syn_mean
                    ci_syn = (coef_syn_mean - 1.96 * np.sqrt(adjusted_var_syn), coef_syn_mean + 1.96 * np.sqrt(adjusted_var_syn))

                    res = [estimate_agreement(ci_init, coef_syn_mean),
                        decision_agreement(coef_init, ci_init, coef_syn_mean, ci_syn),
                        standardized_difference(coef_init, coef_syn_mean, se_init),
                        ci_overlap(ci_init, ci_syn)]

                    # score_df.loc[len(score_df)] = [generator, treatment_effect, m, j] + res
                    score_df.loc[len(score_df)] = [generator, j] + res

    scores = score_df.groupby(['Generator', 'Nb generated datasets'], as_index=False).mean()
    metric_names = scores.columns.values[2:]
    num_metrics = len(metric_names)
    fig, axs = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

    if num_metrics == 1:
        axs = [axs]  # ensure axs is iterable

    for i, ax in enumerate(axs):
        # Format axis spines
        metric_name = metric_names[i]
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        sns.lineplot(data=scores, x='Nb generated datasets', y=metric_name,
                     hue="Generator", ax=ax, palette = 'colorblind')
        ax.set_xlabel('Nb generated datasets', fontsize=20, fontweight="semibold")
        ax.set_ylabel(metric_name, fontsize=20, fontweight="semibold")
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_ylim(0, 1.05)
    plt.tight_layout(pad=3)
    plt.savefig("./dataset/" + dataset_name + "/results_replicability_independent_n_samples_" + str(n_samples) + "n_features_bytype_" + str(n_features_bytype) + ".jpeg")

if __name__ == "__main__":
    run()