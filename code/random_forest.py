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

# %%
# RANDOM FOREST
# Loss function to be optimized
criterion = ['mse', 'mae']

# Number of trees used in the boosting process
n_estimators = [10, 25, 50, 75, 100]

# Maximum depth of each tree
max_depth = [None, 2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'criterion': criterion,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = RandomForestRegressor(random_state = 60)

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

# %%
# Changing n estimators to verify the best performance
# Result:  25 is best
# Create a range of trees to evaluate
trees_grid = {'n_estimators': [10, 25, 50, 75, 100]}

model = RandomForestRegressor(criterion = 'mae',
                                  max_depth = 5,
                                  min_samples_leaf = 8,
                                  min_samples_split = 4,
                                  max_features = 'auto',
                                  random_state = 60)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)


# %%
# Plot the training and testing error vs number of trees
figsize = (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Number of Trees');

# plt.savefig('random_forest/model_change_n_estimators.png')



#%%
# Changing n max_depth to verify the best performance
# Result: 5 is best
# Create a range of trees to evaluate
trees_grid = {'max_depth': [None, 1,2,3,4,5,10,15,20]}

model = RandomForestRegressor(criterion = 'mae',
                                  n_estimators = 25,
                                  min_samples_leaf = 8,
                                  min_samples_split = 4,
                                  max_features = 'auto',
                                  random_state = 60)

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
plt.plot(results['param_max_depth'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_max_depth'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Max Depth'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Max Depth');

# plt.savefig('random_forest/model_change_max_depth.png')


#%%
# Changing n min_samples_leaf to verify the best performance
# Result: 40
# Create a range of trees to evaluate
trees_grid = {'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

model = RandomForestRegressor(criterion = 'mae',
                                  n_estimators = 75,
                                  max_depth = 5,
                                  min_samples_split = 4,
                                  max_features = 'auto',
                                  random_state = 60)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize=(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_min_samples_leaf'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_min_samples_leaf'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Min Sample Leaf'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Min Sample Leaf');
# plt.savefig('random_forest/model_change_min_samples_leaf.png')

#%%
# Changing n min_samples_split to verify the best performance
# Result: no one is best. chosen 2.
# Create a range of trees to evaluate
trees_grid = {'min_samples_split': [2, 4, 6, 8, 10]}

model = RandomForestRegressor(criterion = 'mae',
                                  n_estimators = 75,
                                  max_depth = 5,
                                  min_samples_leaf = 40,
                                  max_features = 'auto',
                                  random_state = 60)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize=(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_min_samples_split'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_min_samples_split'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Min Samples Split'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Min Samples Split');
# plt.savefig('random_forest/model_change_min_samples_split.png')

#%%
# Changing max_features to verify the best performance
# Result: best is 51
# Create a range of trees to evaluate
trees_grid = {'max_features': [51, int(np.sqrt(51)), int(np.log2(51))]}

model = RandomForestRegressor(criterion = 'mae',
                                  n_estimators = 75,
                                  max_depth = 2,
                                  min_samples_leaf = 50,
                                  min_samples_split = 2,
                                  random_state = 60)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize=(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_max_features'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_max_features'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Max features'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Max features');
# plt.savefig('random_forest/model_change_max_features.png')



# %%
