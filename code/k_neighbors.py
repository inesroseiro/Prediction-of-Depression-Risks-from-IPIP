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

#%%
# Number of trees used in the boosting process
n_neighbors = [2, 5, 10, 15, 20, 30, 40, 50]

# Maximum depth of each tree
weights = ['uniform', 'distance']

# Minimum number of samples per leaf
algorithm = ['ball_tree', 'kd_tree', 'brute']

# Minimum number of samples to split a node
p = [1, 2]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'n_neighbors': n_neighbors,
                       'weights': weights,
                       'algorithm': algorithm,
                       'p': p,}

# Create the model to use for hyperparameter tuning
model = KNeighborsRegressor()

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X, y)

# Get all of the cv results and sort by the test performance
random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)

random_results.head(10)

random_cv.best_estimator_

#%%
# Changing leaf_size to verify the best performance
# Result: 15
# Create a range of trees to evaluate
trees_grid = {'leaf_size': [1,2,4,6,8,10,12, 14, 16, 18, 20, 25, 30, 40, 50, 60 ,70, 80, 90, 100]}

model = KNeighborsRegressor(n_neighbors=40,p=2,weights='uniform', algorithm='kd_tree')

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize = (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_leaf_size'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_leaf_size'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Leaf Size'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Leaf Size');

#%%
# Changing algorithm to verify the best performance
# Result: kd_tree
# Create a range of trees to evaluate
trees_grid = {'algorithm': ['ball_tree', 'kd_tree', 'brute']}

model = KNeighborsRegressor(n_neighbors=30,p=1,weights='distance', leaf_size=30)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize = (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_algorithm'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_algorithm'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Leaf Size'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Leaf Size');

#%%
# Changing weights to verify the best performance
# Result: uniform
# Create a range of trees to evaluate
trees_grid = {'weights':  ['uniform', 'distance']}

model = KNeighborsRegressor(n_neighbors=30,p=1,leaf_size=30, algorithm='brute')

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize = (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_weights'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_weights'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Leaf Size'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Leaf Size');

#%%
# Changing n_neighbors to verify the best performance
# Result: 12
# Create a range of trees to evaluate
trees_grid = {'n_neighbors': [5, 10, 15, 20]}

model = KNeighborsRegressor(algorithm='kd_tree')

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize = (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_neighbors'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_n_neighbors'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Leaf Size'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Leaf Size');


#%%
# Changing p to verify the best performance
# Result: 1 best
# Create a range of trees to evaluate
trees_grid = {'p': [1, 1.2, 1.4, 1.6, 1.8, 2]}

model = KNeighborsRegressor(leaf_size=30,n_neighbors=40,weights='uniform', algorithm='brute')

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize = (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_p'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_p'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Leaf Size'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Leaf Size');


# %%
