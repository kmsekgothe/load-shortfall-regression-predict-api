#!/usr/bin/env python
# coding: utf-8

# # Regression Predict Student Solution
# 
# © Explore Data Science Academy
# 
# ---
# ### Honour Code
# 
# I {**YOUR NAME, YOUR SURNAME**}, confirm - by submitting this document - that the solutions in this notebook are a result of my own work and that I abide by the [EDSA honour code](https://drive.google.com/file/d/1QDCjGZJ8-FmJE3bZdIQNwnJyQKPhHZBn/view?usp=sharing).
# 
# Non-compliance with the honour code constitutes a material breach of contract.
# 
# ### Predict Overview: Spain Electricity Shortfall Challenge
# 
# The government of Spain is considering an expansion of it's renewable energy resource infrastructure investments. As such, they require information on the trends and patterns of the countries renewable sources and fossil fuel energy generation. Your company has been awarded the contract to:
# 
# - 1. analyse the supplied data;
# - 2. identify potential errors in the data and clean the existing data set;
# - 3. determine if additional features can be added to enrich the data set;
# - 4. build a model that is capable of forecasting the three hourly demand shortfalls;
# - 5. evaluate the accuracy of the best machine learning model;
# - 6. determine what features were most important in the model’s prediction decision, and
# - 7. explain the inner working of the model to a non-technical audience.
# 
# Formally the problem statement was given to you, the senior data scientist, by your manager via email reads as follow:
# 
# > In this project you are tasked to model the shortfall between the energy generated by means of fossil fuels and various renewable sources - for the country of Spain. The daily shortfall, which will be referred to as the target variable, will be modelled as a function of various city-specific weather features such as `pressure`, `wind speed`, `humidity`, etc. As with all data science projects, the provided features are rarely adequate predictors of the target variable. As such, you are required to perform feature engineering to ensure that you will be able to accurately model Spain's three hourly shortfalls.
#  
# On top of this, she has provided you with a starter notebook containing vague explanations of what the main outcomes are. 

# <a id="cont"></a>
# 
# ## Table of Contents
# 
# <a href=#one>1. Importing Packages</a>
# 
# <a href=#two>2. Loading Data</a>
# 
# <a href=#three>3. Exploratory Data Analysis (EDA)</a>
# 
# <a href=#four>4. Data Engineering</a>
# 
# <a href=#five>5. Modeling</a>
# 
# <a href=#six>6. Model Performance</a>
# 
# <a href=#seven>7. Model Explanations</a>

#  <a id="one"></a>
# ## 1. Importing Packages
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Importing Packages ⚡ |
# | :--------------------------- |
# | In this section you are required to import, and briefly discuss, the libraries that will be used throughout your analysis and modelling. |
# 
# ---

# In[82]:


# Libraries for data loading, data manipulation and data visulisation
import numpy as np #used to evaluate arrays
import pandas as pd #used to create and utilise tabular data ie Pandas DataFrame
import matplotlib.pyplot as plt #used to visualize data
import seaborn as sns #used to visualize data
from matplotlib import rc

# Libraries for data preparation and model building
import sklearn

# Setting global constants to ensure notebook results are reproducible
#PARAMETER_CONSTANT = ###


# <a id="two"></a>
# ## 2. Loading the Data
# <a class="anchor" id="1.1"></a>
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Loading the data ⚡ |
# | :--------------------------- |
# | In this section you are required to load the data from the `df_train` file into a DataFrame. |
# 
# ---

# In[83]:


df = pd.read_csv('df_train.csv', index_col=0) # load the data
pd.set_option('display.max_columns', None)


# In[84]:


#Basic Analysis

df.shape #DataFrame has 8763 rows and 48 columns


# In[85]:


df.head(50)


# In[86]:


print(df['Seville_pressure'].unique())
print(df['Valencia_wind_deg'].unique())


# In[87]:


df.info()


# In[88]:


df['time']=pd.to_datetime(df['time'])


# In[89]:


df.info()


# In[90]:


df.dtypes


# In[91]:


#Extract day, month, year, hour
df['Day'] = df['time'].dt.day


# In[92]:


df['Month'] = df['time'].dt.month


# In[93]:


df['Year'] = df['time'].dt.year


# In[94]:


df["Hour"]=df['time'].dt.hour


# In[95]:


df.drop('time',axis=1,inplace=True)


# In[96]:


df.head(2)


# In[97]:


#Extract the number from Seville_pressure (turn object into numeric datatime)
df['Seville_pressure'] = df['Seville_pressure'].str[2:]
df['Seville_pressure'] = round(df['Seville_pressure'].astype(float), 6)


# In[98]:


df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str[6:]
df['Valencia_wind_deg'] = round(df['Valencia_wind_deg'].astype(float), 6)


# In[99]:


df.head()


# In[100]:


df.dtypes


# In[101]:


df.isnull().sum()


# In[102]:


df = df.sort_index(axis=1)
df.head()


# In[103]:


df.shape


# <a id="three"></a>
# ## 3. Exploratory Data Analysis (EDA)
# <a class="anchor" id="1.1"></a>
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Exploratory data analysis ⚡ |
# | :--------------------------- |
# | In this section, you are required to perform an in-depth analysis of all the variables in the DataFrame. |
# 
# ---
# 

# In[104]:


# look at data statistics

df.describe()
###These are the descriptive statistics of each feature. 
###Notice how, from above, df['Valencia_pressure'] has 2068 null values out of 8763


# In[105]:


#are Barcelona_rain_1h, Barcelona_rain_3h, Bilbao_rain_1h, Bilbao_snow_3h, Madrid_clouds_all, Madrid_rain_1h, 
#Seville_rain_1h, Seville_rain_3h, Valencia_snow_3h categorical data? Why are the min, 25%, 50% and 75% zero?

print(df['Barcelona_rain_1h'].unique())
print(df['Bilbao_rain_1h'].unique())


# In[106]:


skew = df.skew() #skew statistics indicate how symmetrical the data is
df_skew = skew.to_frame('Skew')
skew_list = np.array(df_skew['Skew'])


# In[107]:


def skew_interpretation(slist):
    interpretation = []
    for x in slist:
        if -0.5<x and x<0.5:
            interpretation.append("Fairly Symmetrical")
        elif -1<x and x<-0.5:
            interpretation.append("Moderately Negative")
        elif -1>x:
            interpretation.append("Highly Negative")
        elif 0.5<x and x<=1:
            interpretation.append("Moderately Positive")
        else:
            interpretation.append("Highly Positive")
    return np.array(interpretation)

df_skew['Interpretation'] = skew_interpretation(skew_list)
df_skew


# In[108]:


kurtosis = df.kurtosis() #kurtosis statistics indicate columns have a large number of outliers or lack outliers
df_kurtosis = kurtosis.to_frame('Kurtosis')
kurtosis_list = np.array(df_kurtosis['Kurtosis'])


# In[109]:


def kurtosis_interpretation(klist):
    interpretation = []
    for x in klist:
        if 3<=x:
            interpretation.append("Large Number of Outliers")
        else:
            interpretation.append("Lack of Outliers")
    return np.array(interpretation)

df_kurtosis['Interpretation'] = kurtosis_interpretation(kurtosis_list)
df_kurtosis


# In[110]:


df.shape


# In[111]:


# check for linearity

fig, axs = plt.subplots(13, 4, figsize=(100,150),)
fig.subplots_adjust(hspace=2, wspace=0.2)
axs = axs.ravel()

for index, column in enumerate(df.columns):
    axs[index-1].set_title("{} vs. load_shortfall_3h".format(column), fontsize=16)
    axs[index-1].scatter(x=df[column], y=df['load_shortfall_3h'], color='blue', edgecolor='k')
    
fig.tight_layout(pad=1)


# In[112]:


def plot(df, col):
    fig, (ax1,ax2) = plt.subplots(2,1)
    sns.histplot(df[col], ax=ax1)
    sns.boxplot(df[col], ax=ax2)


# In[113]:


plt.figure(figsize = (30,20))
plot(df, 'Barcelona_rain_1h')


# In[114]:


#Impute Valencia_pressure with the median and not mean because the data is highly negatively skewed
val_pressure = df.Valencia_pressure.median()
df.loc[df.Valencia_pressure.isnull(),'Valencia_pressure'] = val_pressure


# In[115]:


df.isnull().sum()


# In[116]:


# Distributions & Extreme Values
features = df.columns
df[features].plot(kind='density', subplots=True, layout=(7, 8), sharex=False, figsize=(40, 40));


# In[117]:


df.hist(bins=10, figsize=(30,30)) 
plt.show


# In[118]:


#Reorder the columns
column_titles = [col for col in df.columns if col!= 'load_shortfall_3h'] + ['load_shortfall_3h']
df = df.reindex(columns=column_titles)


# In[119]:


# evaluate correlation
plt.figure(figsize = (40,40))
heatmap = sns.heatmap(df.corr(), annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':30}, pad=12);


# In[120]:


print(df['Valencia_wind_deg'].unique())


# In[121]:


plt.figure(figsize=(8,14))
hm = sns.heatmap(df.corr()[['load_shortfall_3h']].sort_values(by='load_shortfall_3h', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
hm.set_title('Features correlating with Load Shortfall', fontdict={'fontsize':18}, pad=16);


# In[122]:


# have a look at feature distributions


# <a id="four"></a>
# ## 4. Data Engineering
# <a class="anchor" id="1.1"></a>
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Data engineering ⚡ |
# | :--------------------------- |
# | In this section you are required to: clean the dataset, and possibly create new features - as identified in the EDA phase. |
# 
# ---

# In[123]:


# remove missing values/ features
df.isnull().sum()


# In[124]:


# create new features
df


# In[125]:


# drop columns where there's high multicolinearity

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range (len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i,j])>=threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[126]:


cols_related = correlation(df, 0.65)
len(cols_related)


# In[127]:


trimmed_df = df.drop(cols_related, axis=1)
trimmed_df.shape


# In[ ]:





# In[128]:


# engineer existing features (Scale features using standardisation)
X = trimmed_df.drop('load_shortfall_3h', axis=1)
y = trimmed_df['load_shortfall_3h']


# In[129]:


from sklearn.preprocessing import StandardScaler


# In[130]:


scale = StandardScaler()


# In[131]:


X_scaled = scale.fit_transform(X)


# In[132]:


X_std = pd.DataFrame(X_scaled, columns=X.columns)
X_std.head()


# In[133]:


# Inspect the new dataframe
X_std.describe().loc['std']


# In[134]:


X_std_copy = X_std.copy()
X_std_copy.describe()


# <a id="five"></a>
# ## 5. Modelling
# <a class="anchor" id="1.1"></a>
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Modelling ⚡ |
# | :--------------------------- |
# | In this section, you are required to create one or more regression models that are able to accurately predict the thee hour load shortfall. |
# 
# ---

# In[135]:


# split the train data further into train/test data (to perform validation before bringing in the true test data)
from sklearn.model_selection import train_test_split


# In[136]:


X_train, X_test, y_train, y_test = train_test_split(X_std_copy, y, test_size=0.2, random_state=50, shuffle=False)


# In[137]:


# Plot the results


# In[138]:


# create targets and features dataset


# In[144]:


# create one or more ML models
# Ridge Regression

from sklearn.linear_model import Ridge
ridge = Ridge()#alpha=0.01, normalize=True)


# In[145]:


ridge.fit(X_train, y_train)


# In[146]:


# evaluate one or more ML models
b0 = float(ridge.intercept_)
print("Intercept:", float(b0))


# In[147]:


coeff = pd.DataFrame(ridge.coef_, X.columns, columns=['Coefficient'])


# In[148]:


coeff.head()


# <a id="six"></a>
# ## 6. Model Performance
# <a class="anchor" id="1.1"></a>
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Model performance ⚡ |
# | :--------------------------- |
# | In this section you are required to compare the relative performance of the various trained ML models on a holdout dataset and comment on what model is the best and why. |
# 
# ---

# In[149]:


# Compare model performance
from sklearn.linear_model import LinearRegression

# Create model object
lm = LinearRegression()

# Train model
lm.fit(X_train, y_train)


# In[150]:


from sklearn import metrics


# In[151]:


train_lm = lm.predict(X_train)
train_ridge = ridge.predict(X_train)

print('Training MSE')
print('Linear:', metrics.mean_squared_error(y_train, train_lm))
print('Ridge :', metrics.mean_squared_error(y_train, train_ridge))


# In[152]:


test_lm = lm.predict(X_test)
test_ridge = ridge.predict(X_test)

print('Testing MSE')
print('Linear:', metrics.mean_squared_error(y_test, test_lm))
print('Ridge :', metrics.mean_squared_error(y_test, test_ridge))


# In[153]:


train_plot = y_train.append(pd.Series(y_test, index=['7010']))


# In[154]:


plt.figure(figsize=(200,15))
plt.plot(np.arange(len(y)), ridge.predict(X_std), label='Predicted')
plt.plot(np.arange(len(train_plot)), train_plot, label='Training')
plt.plot(np.arange(len(y_test))+len(y_train), y_test, label='Testing')
plt.legend()

plt.show()


# In[155]:


# r_squared
print('Training R squared')
print('Linear:', metrics.mean_squared_error(y_train, train_lm)*len(train_lm))
print('Ridge :', metrics.mean_squared_error(y_train, train_ridge)*len(train_ridge))


# In[156]:


# r_squared 
print('Testing R squared')
print('Linear:', metrics.mean_squared_error(y_test, test_lm) * len(test_lm))
print('Ridge :', metrics.mean_squared_error(y_test, test_ridge) * len(test_ridge))


# In[157]:


# RMSE

print('Training RMSE')
print('Linear :', metrics.mean_squared_error(y_train, train_ridge, squared=False))
print('Ridge :', metrics.mean_squared_error(y_test, test_ridge, squared=False))


# In[158]:


print('Testing RMSE')
print('Linear :', metrics.mean_squared_error(y_test, test_ridge, squared=False))
print('Ridge :', metrics.mean_squared_error(y_test, test_ridge, squared=False))


# In[79]:


# Choose best model and motivate why it is the best choice


# <a id="seven"></a>
# ## 7. Model Explanations
# <a class="anchor" id="1.1"></a>
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Model explanation ⚡ |
# | :--------------------------- |
# | In this section, you are required to discuss how the best performing model works in a simple way so that both technical and non-technical stakeholders can grasp the intuition behind the model's inner workings. |
# 
# ---

# In[ ]:


# discuss chosen methods logic

