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
# Loss function to be optimized
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = GradientBoostingRegressor(random_state = 42)

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

random_cv.best_estimator_

#%%
# Changing n estimators to verify the best performance
# Result: 75
# Create a range of trees to evaluate
trees_grid = {'n_estimators': [25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}

model = GradientBoostingRegressor(loss = 'huber', max_depth = 3,
                                  min_samples_leaf = 4,
                                  min_samples_split = 10,
                                  max_features = 'sqrt',
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)



# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
# plt.savefig('gradient_boosting/model_change_n_estimators.png')
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,10))
plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Estimators number'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Estimators number');

#%%
# Changing n max_depth to verify the best performance
# Result: 2 is best
# Create a range of trees to evaluate
trees_grid = {'max_depth': [1,2,3,4,5,10,15,20]}

model = GradientBoostingRegressor(loss = 'huber', n_estimators = 75,
                                  min_samples_leaf = 4,
                                  min_samples_split = 10,
                                  max_features = 'sqrt',
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
plt.style.use('fivethirtyeight')
plt.plot(results['param_max_depth'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_max_depth'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Max depth'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Max depth');
plt.figure(figsize=(15,10))
# plt.savefig('gradient_boosting/model_change_max_depths.png')



#%%
# Changing n min_samples_leaf to verify the best performance
# Result: 10
# Create a range of trees to evaluate
trees_grid = {'min_samples_leaf': [2, 10, 25, 50, 100, 200, 300, 400, 500]}

model = GradientBoostingRegressor(loss = 'huber', n_estimators = 75,
                                  max_depth = 2,
                                  min_samples_split = 10,
                                  max_features = 'sqrt',
                                  random_state = 42)

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
plt.plot(results['param_min_samples_leaf'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_min_samples_leaf'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel(' Min samples leaf'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Min samples leaf');
plt.figure(figsize=(15,10));
# plt.savefig('gradient_boosting/model_change_min_samples_leaf.png')


#%%
# Changing n min_samples_split to verify the best performance
# Result: 10, mas tudo igual
# Create a range of trees to evaluate
trees_grid = {'min_samples_split': [5687, 5688, 5689, 5690, 5691]}

model = GradientBoostingRegressor(loss = 'huber', n_estimators = 75,
                                  max_depth = 2,
                                  min_samples_leaf = 18,
                                  max_features = 'sqrt',
                                  random_state = 42)

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
plt.plot(results['param_min_samples_split'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_min_samples_split'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Min samples split'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Min samples split');
# plt.savefig('gradient_boosting/model_change_min_samples_split.png')


#%%
# Changing max_features to verify the best performance
# Result: 51, mas sqrt(51) debatível
# Create a range of trees to evaluate
trees_grid = {'max_features': [51, 50, 45, 40, 35, 30, 25, 20, 15, 10, int(np.sqrt(51)), int(np.log2(51))]}

model = GradientBoostingRegressor(loss = 'huber', n_estimators = 75,
                                  max_depth = 2,
                                  min_samples_leaf = 18,
                                  min_samples_split = 10,
                                  random_state = 42)

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
plt.plot(results['param_max_features'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_max_features'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Max features'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Max features');
# plt.savefig('gradient_boosting/model_change_max_features.png')
