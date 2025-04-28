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



def log_rank(data_init, data_syn):
    
    # surv_time_init, surv_ind_init  = data_init['time'].values.astype(bool), data_init['censor'].values
    # surv_data_init = Surv.from_arrays(surv_ind_init, surv_time_init)
    # treatment_init = data_init['treatment'].values
    # logrank_init = - np.log(compare_survival(surv_data_init, treatment_init)[1])

    # logrank_syn = []
    # n_syn_dataset = len(data_syn)
    # for i in range(n_syn_dataset):
    #     surv_time_syn, surv_ind_syn  = data_syn[i]['time'].values.astype(bool), data_syn[i]['censor'].values
    #     surv_data_syn = Surv.from_arrays(surv_ind_syn, surv_time_syn)
    #     treatment_syn = data_syn[i]['treatment'].values
    #     logrank_syn.append(-np.log(compare_survival(surv_data_syn, treatment_syn)[1]))

    control_init = data_init[data_init['treatment'] == 0]
    treat_init = data_init[data_init['treatment'] == 1]
    surv_time_control_init  = control_init['time'].values.astype(bool)
    surv_ind_control_init = control_init['censor'].values
    surv_time_treat_init  = treat_init['time'].values.astype(bool)
    surv_ind_treat_init = treat_init['censor'].values
    res_init = logrank_test(surv_time_control_init, surv_time_treat_init, 
                           event_observed_A=surv_ind_control_init, 
                           event_observed_B=surv_ind_treat_init)
    logrank_init = -np.log(res_init.p_value)

    logrank_syn = []
    n_syn_dataset = len(data_syn)
    for i in range(n_syn_dataset):
        control_syn = data_syn[i][data_syn[i]['treatment'] == 0]
        treat_syn = data_syn[i][data_syn[i]['treatment'] == 1]
        surv_time_control_syn  = control_syn['time'].values.astype(bool)
        surv_ind_control_syn = control_syn['censor'].values
        surv_time_treat_syn  = treat_syn['time'].values.astype(bool)
        surv_ind_treat_syn = treat_syn['censor'].values
        res_init = logrank_test(surv_time_control_syn, surv_time_treat_syn, 
                            event_observed_A=surv_ind_control_syn, 
                            event_observed_B=surv_ind_treat_syn)
        logrank_syn.append(-np.log(res_init.p_value))

    return logrank_init, np.array(logrank_syn)

def strata_log_rank(data_init, data_syn, strata=None):

    surv_time_init, surv_ind_init  = data_init['time'].values, data_init['censor'].values.astype(bool)
    treatment_init = data_init['treatment'].values
    strata_init = data_init[strata].values
    res_init = multivariate_logrank_test(surv_time_init, treatment_init, surv_ind_init, strata=strata_init)
    logrank_init = -np.log(res_init.p_value)

    logrank_syn = []
    n_syn_dataset = len(data_syn)
    for i in range(n_syn_dataset):
        surv_time_syn, surv_ind_syn  = data_syn[i]['time'].values, data_syn[i]['censor'].values.astype(bool)
        treatment_syn = data_syn[i]['treatment'].values
        strata_syn = data_syn[i][strata].values
        res_syn = multivariate_logrank_test(surv_time_syn, treatment_syn, surv_ind_syn, strata=strata_syn)
        logrank_syn.append(-np.log(res_syn.p_value))

    return logrank_init, np.array(logrank_syn)

def cox_estimation(data_init, data_syn):

    # surv_time_init, surv_ind_init  = data_init['time'].values, data_init['censor'].values.astype(bool)
    # surv_data_init = Surv.from_arrays(surv_ind_init, surv_time_init)
    # treatment_init = data_init['treatment'].values.reshape(-1, 1)
    # cox_init = CoxPHSurvivalAnalysis()
    # cox_init.fit(treatment_init, surv_data_init)
    # coef_init = cox_init.coef_[:]

    # coef_syn = []
    # n_syn_dataset = len(data_syn)
    # for i in range(n_syn_dataset):
    #     surv_time_syn, surv_ind_syn  = data_syn[i]['time'].values.astype(bool), data_syn[i]['censor'].values
    #     surv_data_syn = Surv.from_arrays(surv_ind_syn, surv_time_syn)
    #     treatment_syn = data_syn[i]['treatment'].values.reshape(-1, 1)
    #     cox_syn = CoxPHSurvivalAnalysis()
    #     cox_syn.fit(treatment_syn, surv_data_syn)
    #     coef_syn.append(cox_syn.coef_[:])
    cph = CoxPHFitter()
    cph.fit(data_init[['time', 'censor', 'treatment']], 
            duration_col='time', event_col='censor')
    coef_init = cph.summary.coef.values
    p_value_init = cph.summary.p.values

    coef_syn = []
    p_value_syn = []
    n_syn_dataset = len(data_syn)
    for i in range(n_syn_dataset):
        cph = CoxPHFitter()
        cph.fit(data_syn[i][['time', 'censor', 'treatment']], 
                duration_col='time', event_col='censor')
        coef_syn.append(cph.summary.coef.values)
        p_value_syn.append(cph.summary.p.values)

    return coef_init, np.array(coef_syn), p_value_init, np.array(p_value_syn)

def strata_cox_estimation(data_init, data_syn, strata=None):

    # surv_time_init, surv_ind_init  = data_init['time'].values, data_init['censor'].values.astype(bool)
    # surv_data_init = Surv.from_arrays(surv_ind_init, surv_time_init)
    # treatment_init = data_init['treatment'].values.reshape(-1, 1)
    # cox_init = CoxPHSurvivalAnalysis()
    # cox_init.fit(treatment_init, surv_data_init)
    # coef_init = cox_init.coef_[:]

    # coef_syn = []
    # n_syn_dataset = len(data_syn)
    # for i in range(n_syn_dataset):
    #     surv_time_syn, surv_ind_syn  = data_syn[i]['time'].values.astype(bool), data_syn[i]['censor'].values
    #     surv_data_syn = Surv.from_arrays(surv_ind_syn, surv_time_syn)
    #     treatment_syn = data_syn[i]['treatment'].values.reshape(-1, 1)
    #     cox_syn = CoxPHSurvivalAnalysis()
    #     cox_syn.fit(treatment_syn, surv_data_syn)
    #     coef_syn.append(cox_syn.coef_[:])
    cph = CoxPHFitter()
    cph.fit(data_init[['time', 'censor', 'treatment'] + [strata]], 
            duration_col='time', event_col='censor', strata=[strata])
    coef_init = cph.summary.coef.values
    p_value_init = cph.summary.p.values

    coef_syn = []
    p_value_syn = []
    n_syn_dataset = len(data_syn)
    for i in range(n_syn_dataset):
        cph = CoxPHFitter()
        cph.fit(data_syn[i][['time', 'censor', 'treatment'] + [strata]], 
                duration_col='time', event_col='censor', strata=[strata])
        coef_syn.append(cph.summary.coef.values)
        p_value_syn.append(cph.summary.p.values)

    return coef_init, np.array(coef_syn), p_value_init, np.array(p_value_syn)