import pandas as pd
import acquire 


def prep_iris():

    df_iris = acquire.get_iris_data()
    
    df_iris = df_iris.drop(columns = ['species_id', 'measurement_id', 'Unnamed: 0'])
    
    df_iris = df_iris.rename(columns={"species_name": "species"})
    
    df_dummy = pd.get_dummies(df_iris[['species']])
    
    return pd.concat([df_iris, df_dummy], axis = 1)

