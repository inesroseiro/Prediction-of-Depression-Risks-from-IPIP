#%%
import pandas as pd
from sklearn.model_selection import train_test_split

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# LIME for explaining predictions
import lime
import lime.lime_tabular

from sklearn import tree


# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Matplotlib for visualization
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Imputing missing values and scaling values
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer;
# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


df = pd.read_csv("data/dataIPIP.csv", header=None,error_bad_lines=False)

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

#missing_values_table(df); nao existe missing values.

new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header

del df["ID"]

# Iterate through the columns
for col in list(df.columns):
    # Select columns that should be numeric - every column 
    # Convert the data type to float
    df[col] = df[col].astype(float)


# Remove outliers
# Calculate first and third quartile

first_quartile =  df['ND8MAL'].describe()['25%']
third_quartile =  df['ND8MAL'].describe()['75%']

# Interquartile range
iqr = third_quartile - first_quartile

# Remove outliers
df = df[(df['ND8MAL'] > (first_quartile - 3 * iqr)) &
            (df['ND8MAL'] < (third_quartile + 3 * iqr))]



df = df[df['ND8MAL'] != -2]

print(df.shape)


# Separate input features (X) and target variable (y)
features = df.drop(columns='ND8MAL')
targets = pd.DataFrame(df['ND8MAL'])


# Split into 80% training and 20% testing set
X, X_test, Y, Y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

# save data to csv 
X.to_csv('data/training_features.csv')
X_test.to_csv('data/testing_features.csv')
Y.to_csv('data/training_labels.csv')
Y_test.to_csv('data/testing_labels.csv')

# %%
