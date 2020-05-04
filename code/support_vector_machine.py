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
# Changing kernel to verify the best performance
# Result: best is rbf
# Create a range of trees to evaluate
trees_grid = {'kernel': ['rbf', 'linear', 'poly']}

model = SVR(C=400, epsilon=0.1)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_kernel'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_kernel'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('kernel'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs kernel');

#%%
# Changing C to verify the best performance
# Result: best is 500
# Create a range of trees to evaluate
trees_grid = {'C': [1.0, 25, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000]}

model = SVR(kernel='rbf', epsilon=0.1)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
# %%
figsize =(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_C'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_C'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('C value'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs C value');
#plt.savefig('support_vector_machine/model_change_cval.png')


#%%
# Now SUPPORT VECTOR MACHINE
# Changing epsilon to verify the best performance
# Result: praticamente insignificante. best is 0.01
# Create a range of trees to evaluate
trees_grid = {'epsilon': [0.00001, 0.00001, 0.0001, 0.001, 0.01]}

model = SVR(kernel='rbf',C=500)

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
plt.plot(results['param_epsilon'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_epsilon'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Epsilon'); plt.ylabel('Mean Absolute Error'); plt.legend();
plt.title('Performance vs Epsilon');
# plt.savefig('support_vector_machine/model_change_epsilon.png')



# %%
