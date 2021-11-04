import numpy as np
import pandas as pd

# Libraries for data preparation and model building
#import matplotlib.pyplot as plt
#import seaborn as sns

df_test = pd.read_csv('df_test.csv')

df_clean = df_test
median_val = df_clean['Valencia_pressure'].median()
df_clean['Valencia_pressure'] = df_clean['Valencia_pressure'].fillna(df_clean['Valencia_pressure'].median())
df_clean['Valencia_pressure'] = df_clean['Valencia_pressure'].replace(np.nan,df_clean['Valencia_pressure'].median())
#df_clean.loc[df_clean['Valencia_pressure'].isnull()] = median_val


df_clean['time'] = pd.to_datetime(df_clean['time'])

df_clean['Valencia_wind_deg'] = df_clean['Valencia_wind_deg'].str.extract('(\d+)')

df_clean['Valencia_wind_deg'] = pd.to_numeric(df_clean['Valencia_wind_deg'])

df_clean.Seville_pressure = df_clean.Seville_pressure.str.extract('(\d+)')

df_clean.Seville_pressure = pd.to_numeric(df_clean.Seville_pressure)

df_clean = df_clean.drop(['Unnamed: 0' , 'time'], axis = 1)

cols = list(df_clean.columns)
i = 0
for col in cols:
    
    df_clean[col] = df_clean[col].round(2)
for col in cols:
        #df_clean[col] = df_clean[col].round(2)
        for item in df_clean[col]:
            if item == np.nan:
                print(col)
    
#print(df_clean['time'].head(20))
#predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
predict_vector = df_clean
print(predict_vector.shape)
print(predict_vector.columns)
# ------------------------------------------------------------------------
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(predict_vector)
    