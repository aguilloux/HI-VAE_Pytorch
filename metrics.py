#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.compare import compare_survival
from sksurv.util import Surv

from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import CoxPHFitter

from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.metrics.eval import Metrics


def compute_logrank_test(control, treat):
    """
    Perform a two-sample log-rank test comparing the survival distributions
    of control and treatment groups.

    Args:
        control (DataFrame): Subset of the dataset where treatment == 0.
        treat (DataFrame): Subset of the dataset where treatment == 1.

    Returns:
        float: Negative logarithm of the p-value from the log-rank test.
    """
    surv_time_control = control['time'].values.astype(bool)
    surv_event_control = control['censor'].values
    surv_time_treat = treat['time'].values.astype(bool)
    surv_event_treat = treat['censor'].values

    result = logrank_test(
        surv_time_control, surv_time_treat,
        event_observed_A=surv_event_control,
        event_observed_B=surv_event_treat
    )
    return -np.log(result.p_value)

def log_rank(data_init, data_syn):
    """
    Evaluate the difference in survival distributions between treatment and control
    groups for both initial and synthetic datasets using the log-rank test.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.

    Returns:
        tuple: Log-rank test statistic for initial data and array of statistics for synthetic data.
    """
    control_init = data_init[data_init['treatment'] == 0]
    treat_init = data_init[data_init['treatment'] == 1]
    logrank_init = compute_logrank_test(control_init, treat_init)

    logrank_syn = [
        compute_logrank_test(
            data[data['treatment'] == 0],
            data[data['treatment'] == 1]
        ) for data in data_syn
    ]

    return logrank_init, np.array(logrank_syn)


def compute_multivariate_logrank_test(surv_time, treatment, surv_event, strata):
    """
    Perform a stratified log-rank test across specified strata.

    Args:
        surv_time (array): Array of survival times.
        treatment (array): Array indicating treatment group.
        surv_event (array): Event indicator array.
        strata (array): Stratification variable.

    Returns:
        float: Negative logarithm of the p-value from the stratified log-rank test.
    """
    result = multivariate_logrank_test(surv_time, treatment, surv_event, strata=strata)
    return -np.log(result.p_value)

def strata_log_rank(data_init, data_syn, strata):
    """
    Evaluate stratified survival difference between groups on initial and synthetic datasets.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.
        strata (str): Column name to stratify on.

    Returns:
        tuple: Stratified log-rank test statistic for initial data and array for synthetic data.
    """
    surv_time_init, surv_event_init = data_init['time'], data_init['censor'].astype(bool)
    logrank_init = compute_multivariate_logrank_test(
        surv_time_init,
        data_init['treatment'],
        surv_event_init,
        data_init[strata]
    )

    logrank_syn = [
        compute_multivariate_logrank_test(
            data['time'],
            data['treatment'],
            data['censor'].astype(bool),
            data[strata]
        ) for data in data_syn
    ]

    return logrank_init, np.array(logrank_syn)

def fit_cox_model(data, columns, strata=None):
    """
    Fit a Cox proportional hazards model optionally stratified by a variable.

    Args:
        data (DataFrame): Dataset containing survival and covariate information.
        columns (list): List of column names to include in the model.
        strata (list, optional): Stratification variable(s).

    Returns:
        tuple: Coefficients and p-values from the Cox model.
    """
    cph = CoxPHFitter()
    fit_args = {'duration_col': 'time', 'event_col': 'censor'}
    if strata:
        fit_args['strata'] = strata

    cph.fit(data[columns], **fit_args)
    return cph.summary.coef.values, cph.summary.p.values

def cox_estimation(data_init, data_syn):
    """
    Estimate Cox model coefficients and p-values for initial and synthetic datasets.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.

    Returns:
        tuple: Initial coefficients, synthetic coefficients, initial p-values, synthetic p-values.
    """
    columns = ['time', 'censor', 'treatment']
    coef_init, p_value_init = fit_cox_model(data_init, columns)

    results = [fit_cox_model(data, columns) for data in data_syn]
    coef_syn, p_value_syn = zip(*results)

    return coef_init, np.array(coef_syn), p_value_init, np.array(p_value_syn)

def strata_cox_estimation(data_init, data_syn, strata=None):
    """
    Estimate stratified Cox model coefficients and p-values for initial and synthetic datasets.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.
        strata (str): Column to use for stratification.

    Returns:
        tuple: Initial coefficients, synthetic coefficients, initial p-values, synthetic p-values.
    """
    columns = ['time', 'censor', 'treatment', strata]
    coef_init, p_value_init = fit_cox_model(data_init, columns, strata=[strata])

    results = [fit_cox_model(data, columns, strata=[strata]) for data in data_syn]
    coef_syn, p_value_syn = zip(*results)

    return coef_init, np.array(coef_syn), p_value_init, np.array(p_value_syn)

def general_metrics(data_init, data_gen, generator):
    """
    Compute a set of general quality metrics to assess synthetic survival data.

    Args:
        data_init (DataFrame): Initial real-world dataset.
        data_gen (list of DataFrame): List of generated synthetic datasets.
        generator (str): Name of the synthetic data generator.

    Returns:
        DataFrame: Summary of metric scores for each synthetic dataset.
    """

    synthcity_dataloader_init = SurvivalAnalysisDataLoader(data_init, target_column = "censor", time_to_event_column = "time")
    scores = []
    for idx, generated_data in enumerate(data_gen):
        enable_reproducible_results(idx)
        clear_cache()
        synthcity_dataloader_syn = SurvivalAnalysisDataLoader(generated_data, target_column = "censor", time_to_event_column = "time")

        evaluation = Metrics().evaluate(X_gt=synthcity_dataloader_init, # can be dataloaders or dataframes
                                        X_syn=synthcity_dataloader_syn, 
                                        reduction='mean', # default mean
                                        n_histogram_bins=10, # default 10
                                        metrics=None, # all metrics
                                        task_type='survival_analysis', 
                                        use_cache=True)

        selected_metrics = evaluation.T[["stats.jensenshannon_dist.marginal",
                                          "stats.ks_test.marginal", 
                                          "stats.survival_km_distance.optimism", 
                                          "detection.detection_xgb.mean", 
                                          "sanity.nearest_syn_neighbor_distance.mean", 
                                          "privacy.k-map.score"]].T["mean"].values
        scores.append(selected_metrics)

    score_df = pd.DataFrame(scores, columns=["J-S distance", "KS test", "Survival curves distance", 
                                             "Detection XGB", "NNDR", "K-map score"])
    score_df["generator"] = generator

    return score_df