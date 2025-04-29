import pandas as pd
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins

def run(data, columns, target_column, time_to_event_column, n_generated_sample):
    """
    Use a conditional GAN for survival data generation
    """
    
    # Define data and model
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    data = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    model_survgan = Plugins().get("survival_gan") 

    # Train
    cond = df[[target_column]]
    model_survgan.fit(data, cond=cond)
    
    # Generate
    cond_gen = data[[target_column]]
    est_data_gen_transformed_survgan = []
    for j in range(n_generated_sample):
        out = model_survgan.generate(count=577, cond=cond_gen)
        est_data_gen_transformed_survgan.append(out)

    return est_data_gen_transformed_survgan
    