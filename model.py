"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
from numpy.lib import type_check
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    import random
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    df_clean = feature_vector_df
    #Reorder the columns
    column_titles = [col for col in df_clean.columns if col!= 'load_shortfall_3h'] + ['load_shortfall_3h']
    df = df_clean.reindex(columns=column_titles)
    
    cols = list(df_clean.columns)

    

    df_clean['time']=pd.to_datetime(df_clean['time'])

    df_clean['Day'] = df_clean['time'].dt.day # Extract the Day
    df_clean['Month'] = df_clean['time'].dt.month # Extract the Month
    df_clean['Year'] = df_clean['time'].dt.year # Extract the Year
    df_clean["Hour"]= df_clean['time'].dt.hour # Extract the Hour

    df_clean['Valencia_wind_deg'] = df_clean['Valencia_wind_deg'].str.extract('(\d+)')
    df_clean['Valencia_wind_deg'] = pd.to_numeric(df_clean['Valencia_wind_deg'])
    df_clean['Seville_pressure'] = df_clean['Seville_pressure'].str[2:]
    df_clean.Seville_pressure = pd.to_numeric(df_clean.Seville_pressure)

    ls = ['Valencia_wind_deg','Valencia_pressure','Valencia_snow_3h']
    

    median_val = df_clean['Valencia_pressure'].median()
    df_clean['Valencia_pressure'] = df_clean['Valencia_pressure'].fillna(df_clean['Valencia_pressure'].median())
    
    df_clean['Valencia_pressure'] = df_clean['Valencia_pressure'].replace(np.nan,df_clean['Valencia_pressure'].median())
    
    for col in cols:
        try :
            df_clean[col] = df_clean.astype(np.float64)
        except:
            df_clean[col] = [100.0 for item in df_clean[col]]

    from sklearn.feature_selection import SelectKBest

    X = df_clean[[item for item in cols if item != 'load_shortfall_3h']]
    y = df_clean.iloc[:, [-1]]

    selector = SelectKBest(k=38)
    fit = selector.fit(X,y)
    X_best = selector.fit_transform(X,y)

    # Created dataframes showing top columns and their scores
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns) 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

    # naming the dataframe columns
    featureScores.columns = ['Feature','Score']  
    # get best features
    best_cols = ['Valencia_snow_3h','Barcelona_pressure','Bilbao_snow_3h','Bilbao_rain_1h','Seville_rain_1h','Valencia_wind_speed',
    'Madrid_weather_id','Barcelona_rain_3h','Seville_weather_id','Barcelona_wind_speed','Month','Seville_clouds_all',
    'Madrid_wind_speed','Bilbao_wind_speed','Bilbao_weather_id','Valencia_wind_deg','Madrid_pressure',
    'Seville_wind_speed','Bilbao_clouds_all','Seville_temp_max','Barcelona_wind_deg','Hour','Barcelona_rain_1h',
    'Barcelona_weather_id','Valencia_pressure','Valencia_humidity','Seville_humidity','Valencia_temp_max',
    'Day','Bilbao_temp_max','Valencia_temp','Bilbao_temp_min','Bilbao_temp','Year','Barcelona_temp_max','Valencia_temp_min',
    'Barcelona_temp','Madrid_temp_max']
    Top_features= featureScores.nlargest(38,'Score')
    
    best_features = list(Top_features['Feature'])
    # Using the KNeighbors Regressor we attain prediction scores for each feature
    # we then choose the top features that best predict load shortfall 3h
    # these best features list or best cols list stored here lists all which
    # we use to make our prediction
    

    df_clean = df_clean.drop(['Unnamed: 0' , 'time'], axis = 1)    
    
    predict_vector = df_clean[best_cols]

    print(predict_vector.shape)
    print(predict_vector.columns)
    # ------------------------------------------------------------------------
    # ========================================================================

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
