import pandas as pd
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins

def run(data, columns, target_column, time_to_event_column, n_generated_sample):
    """
    Use a VAE for tabular data generation
    """
    
    # Define data and model
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    data = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    model_survae = Plugins().get("survae") 

    # Train
    model_survae.fit(data)
    
    # Generate
    est_data_gen_transformed_survae = []
    for j in range(n_generated_sample):
        out = model_survae.generate(count=data.shape[0])
        est_data_gen_transformed_survae.append(out)

    return est_data_gen_transformed_survae