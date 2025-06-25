import pandas as pd
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.optuna_sample import suggest_all
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.metrics.eval import Metrics
from sklearn.model_selection import KFold
from utils import metrics
import numpy as np
import optuna
import os
import random
import torch

def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)

def run(data, columns, target_column, time_to_event_column, n_generated_dataset, n_generated_sample=None, params=None):
    """
    Use a VAE for tabular data generation
    """

    set_seed()

    # Define data and model
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    data = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    
    if params is not None:
        model = type(Plugins().get("survae"))
        model_survae = model(**params)
    else:
        model_survae = Plugins().get("survae") 
    
    # Train
    model_survae.fit(data)
    
    # Generate
    if isinstance(n_generated_sample, list):
        est_data_gen_transformed_survae_list = []
        for n_generated_sample_ in n_generated_sample:
            est_data_gen_transformed_survae = []
            for j in range(n_generated_dataset):
                out = model_survae.generate(count=n_generated_sample_)
                est_data_gen_transformed_survae.append(out)
            est_data_gen_transformed_survae_list.append(est_data_gen_transformed_survae)

        return est_data_gen_transformed_survae_list
    else:
        if n_generated_sample is None:
            n_generated_sample = data.shape[0]
        est_data_gen_transformed_survae = []
        for j in range(n_generated_dataset):
            out = model_survae.generate(count=n_generated_sample)
            est_data_gen_transformed_survae.append(out)

        return est_data_gen_transformed_survae



def optuna_hyperparameter_search(data, columns, target_column, time_to_event_column, n_generated_dataset, n_splits, n_trials, study_name='optuna_study_survae', metric='survival_km_distance', method=''):
    
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
 
    def objective(trial: optuna.Trial):
        set_seed()
        model_survae = type(Plugins().get("survae"))
        hp_space = model_survae.hyperparameter_space()
        # hp_space[0].high = 100  # speed up for now
        hp_space[3].choices = [1e-3, 1e-4, 1e-5]
        hp_space[4].choices = [64, 128, 200, 256, 512]
        params = suggest_all(trial, hp_space)
        ID = f"trial_{trial.number}"
        print(ID)
        scores = []
        try:
            if method == 'train_full_gen_full':
                full_data_loader = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
                model_survae_trial = model_survae(**params)
                # train on full data
                model_survae_trial.fit(full_data_loader)
                for j in range(n_generated_dataset):
                    # generate as many data as in the all dataset
                    gen_data = model_survae_trial.generate(count=df.shape[0])
                    df_gen_data = gen_data.dataframe()
                    if metric == 'log_rank_test':
                        score_j = metrics.compute_logrank_test(df, df_gen_data)
                    else: # 'survival_km_distance'
                        clear_cache()
                        evaluation = Metrics().evaluate(X_gt=full_data_loader, # can be dataloaders or dataframes
                                                        X_syn=gen_data, 
                                                        reduction='mean', # default mean
                                                        n_histogram_bins=10, # default 10
                                                        n_folds=1,
                                                        metrics={'stats': ['survival_km_distance']},
                                                        task_type='survival_analysis', 
                                                        use_cache=True)
                        score_j = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
                    scores.append(score_j)
            else:
                # k-fold cross-validation
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                for train_index, test_index in kf.split(df):
                    train_data, test_data = df.iloc[train_index], df.iloc[test_index]
                    train_data_loader = SurvivalAnalysisDataLoader(train_data, target_column=target_column, time_to_event_column=time_to_event_column)
                    test_data_loader = SurvivalAnalysisDataLoader(test_data, target_column=target_column, time_to_event_column=time_to_event_column)
                    full_data_loader = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
                    model_survae_trial = model_survae(**params)
                    
                    if method == 'train_train_gen_full':
                        # train on train data
                        model_survae_trial.fit(train_data_loader)
                        score_k = []
                        for j in range(n_generated_dataset):
                            # generate as many data as in the all dataset
                            gen_data = model_survae_trial.generate(count=df.shape[0])
                            df_gen_data = gen_data.dataframe()
                            if metric == 'log_rank_test':
                                score_kj = metrics.compute_logrank_test(df, df_gen_data)
                            else: # 'survival_km_distance'
                                clear_cache()
                                evaluation = Metrics().evaluate(X_gt=full_data_loader, # can be dataloaders or dataframes
                                                                X_syn=gen_data, 
                                                                reduction='mean', # default mean
                                                                n_histogram_bins=10, # default 10
                                                                n_folds=1,
                                                                metrics={'stats': ['survival_km_distance']},
                                                                task_type='survival_analysis', 
                                                                use_cache=True)
                                score_kj = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
                            score_k.append(score_kj)
                    else:
                        # method ='train_train_gen_test':
                        # train on train data
                        model_survae_trial.fit(train_data_loader)
                        score_k = []
                        for j in range(n_generated_dataset):
                            # generate as many data as in the test set
                            gen_data = model_survae_trial.generate(count=test_data.shape[0])
                            df_gen_data = gen_data.dataframe()
                            if metric == 'log_rank_test':
                                score_kj = metrics.compute_logrank_test(test_data, df_gen_data)
                            else: # 'survival_km_distance'
                                clear_cache()
                                evaluation = Metrics().evaluate(X_gt=test_data_loader, # can be dataloaders or dataframes
                                                                X_syn=gen_data, 
                                                                reduction='mean', # default mean
                                                                n_histogram_bins=10, # default 10
                                                                n_folds=1,
                                                                metrics={'stats': ['survival_km_distance']},
                                                                task_type='survival_analysis', 
                                                                use_cache=True)
                                score_kj = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
                            score_k.append(score_kj)
                        
                    scores.append(np.mean(score_k))
            print(f"Score: {np.mean(scores)}")
        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print(params)
            raise optuna.TrialPruned()
        return np.mean(scores)

    
    db_file = study_name + '.db'
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+study_name+'.db')
    else: 
        sampler = optuna.samplers.TPESampler(seed=10)
        study = optuna.create_study(direction="minimize", study_name=study_name, storage='sqlite:///'+study_name+'.db', sampler=sampler)
        default_params = {'n_iter': 1000, 
                          'lr': 1e-3, 
                          'decoder_n_layers_hidden': 3, 
                          'weight_decay': 1e-5,
                          'batch_size': 200, 
                          'n_units_embedding': 500, 
                          'decoder_n_units_hidden': 500, 
                          'decoder_nonlin': 'leaky_relu', 
                          'decoder_dropout': 0, 
                          'encoder_n_layers_hidden': 3, 
                          'encoder_n_units_hidden': 500, 
                          'encoder_nonlin': 'leaky_relu',
                          'encoder_dropout': 0.1}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
    study.optimize(objective, n_trials=n_trials)
    study.best_params  

    return study.best_params, study


def get_n_hyperparameters(generator_name):
    """
    Returns the number of hyperparameters for the SurVAE model.
    """
    model = type(Plugins().get("survae"))
    hp_space = model.hyperparameter_space()
    return len(hp_space)