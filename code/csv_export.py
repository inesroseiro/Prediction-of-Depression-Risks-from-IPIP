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

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_predict(model):
    # Train the model
    model.fit(X, y)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    
    # Return the performance metric
    return model_pred

def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X, y)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae

lr = LinearRegression()
lr_mae = fit_and_predict(lr)
print("LR DEU %f", fit_and_evaluate(lr))
pd.Series(lr_mae).to_csv('lr_pred.csv')

gb = GradientBoostingRegressor(n_estimators = 75, random_state=42, max_depth=2, min_samples_leaf=10, min_samples_split=10, max_features=51)
gb_mae = fit_and_predict(gb)
print("GB DEU %f", fit_and_evaluate(gb))
pd.Series(gb_mae).to_csv('gb_pred.csv')

rf = RandomForestRegressor(criterion='mae', n_estimators=25, max_depth=5, min_samples_split=2, min_samples_leaf=40, max_features=51, random_state=60)
rf_mae = fit_and_predict(rf)
print("RF DEU %f", fit_and_evaluate(rf))
pd.Series(rf_mae).to_csv('rf_pred.csv')

kn = KNeighborsRegressor(n_neighbors=12, weights='uniform', algorithm='kd_tree', leaf_size=15, p=1)
kn_mae = fit_and_predict(kn)
print("KN DEU %f", fit_and_evaluate(kn))
pd.Series(kn_mae).to_csv('kn_pred.csv')


svr = SVR(kernel='rbf',C=500, epsilon=0.01)
svr_mae = fit_and_predict(svr)
print("SVR DEU %f", fit_and_evaluate(svr))
pd.Series(svr_mae).to_csv('svr_pred.csv')



pd.Series(y_test).to_csv('target.csv')

