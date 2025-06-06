import pandas as pd
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.optuna_sample import suggest_all
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.metrics.eval import Metrics
from sklearn.model_selection import KFold
import numpy as np
import optuna

def run(data, columns, target_column, time_to_event_column, n_generated_sample, params=None):
    """
    Use a conditional GAN for survival data generation
    """
    
    # Define data and model
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    data = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    
    if params is not None:
        model = type(Plugins().get("survival_gan"))
        model_survgan = model(**params)
    else:
        model_survgan = Plugins().get("survival_gan") 
        print(model_survgan.__dict__)

    # Train
    cond = df[[target_column]]
    model_survgan.fit(data, cond=cond)
    
    # Generate
    cond_gen = data[[target_column]]
    est_data_gen_transformed_survgan = []
    for j in range(n_generated_sample):
        out = model_survgan.generate(count=data.shape[0], cond=cond_gen)
        est_data_gen_transformed_survgan.append(out)

    return est_data_gen_transformed_survgan
    


def optuna_hyperparameter_search(data, columns, target_column, time_to_event_column, n_generated_sample, n_splits, n_trials, study_name='optuna_study_surv_gan'):
    
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
 
    def objective(trial: optuna.Trial):
        model = type(Plugins().get("survival_gan"))
        hp_space = model.hyperparameter_space()
        hp_space[0].high = 3  # speed up for now
        params = suggest_all(trial, hp_space)
        model_trial = model(**params)
        ID = f"trial_{trial.number}"
        print(ID)
        scores = []
        try:
            # k-fold cross-validation 
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(df):
                train_data, test_data = df.iloc[train_index], df.iloc[test_index]
                train_data_loader = SurvivalAnalysisDataLoader(train_data, target_column=target_column, time_to_event_column=time_to_event_column)
                test_data_loader = SurvivalAnalysisDataLoader(test_data, target_column=target_column, time_to_event_column=time_to_event_column)
                cond = train_data[[target_column]]
                model_trial.fit(train_data_loader, cond=cond)
                # Generate
                cond_gen = test_data[[target_column]]
                score_k = []
                for j in range(n_generated_sample):
                    gen_data = model_trial.generate(count=test_data.shape[0], cond=cond_gen)
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=test_data_loader, # can be dataloaders or dataframes
                                                    X_syn=gen_data, 
                                                    reduction='mean', # default mean
                                                    n_histogram_bins=10, # default 10
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
        
    study = optuna.create_study(direction="minimize", study_name=study_name, storage='sqlite:///'+study_name+'.db')
    study.optimize(objective, n_trials=n_trials)
    study.best_params  

    return study.best_params, study