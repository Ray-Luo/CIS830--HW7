# Project Template
# ~~~~~~~~~~~~~~~~1. Prepare Problem~~~~~~~~~~~~~~~~~~~~~~
# a) Load libraries
# b) Load dataset

# define task: regression on cpu usage

# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from transform import scale_standardize

# load dataset
filename = 'new_dataset_Mem.csv'
dataset = read_csv(filename, header = 0)


# define variables
array = dataset.values
X = array[:,0:7]
Y = array[:,7]
Z = numpy.ones((10001,8))
# transform data
transformedX = scale_standardize(X)
for i in range(len(transformedX)):
  Z[i] = numpy.append(transformedX[i],Y[i])

print(Z[0:5,:])

# ~~~~~~~~~delete 5% and 95% ~~~~~~~~~~~~~~~~

delete_row = list()

row_list = list()

for i in range(len(Z[0])):
  row_list.append([Z[j][i] for j in range(len(Z))])


for i in range(len(row_list)):
  up_percentile = numpy.percentile(row_list[i],95)
  low_percentile = numpy.percentile(row_list[i],5)
  print up_percentile, low_percentile
  for j in range(len(row_list[0])):
    if row_list[i][j] < low_percentile or row_list[i][j] > up_percentile:
      delete_row.append(j)

Z = numpy.delete(Z,delete_row, axis=0)

dataset = DataFrame(Z)


# ~~~~~~~~~~~~~~~~~~2. Summarize Data~~~~~~~~~~~~~~~~~~~~~~
# a) Descriptive statistics
# b) Data visualizations

# shape
print(dataset.shape)

# types
print(dataset.dtypes)

# descriptions
#set_option('precision', 1)
print(dataset.describe())

# correlation
#set_option('precision', 2)
print(dataset.corr(method = 'pearson'))

# histograms
#dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
#pyplot.show()

# density
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False, legend=True, fontsize=1)
pyplot.show()

# scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()






# ~~~~~~~~~~~~~~~~~~~3. Prepare Data~~~~~~~~~~~~~~~~~~~~~
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms




# ~~~~~~~~~~~~~~~~~~~4. Evaluate Algorithms~~~~~~~~~~~~~~~
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms

# SPlit-out validation dataset
X = Z[:,0:7]
Y = Z[:,7]


# feature extraction
#model = LogisticRegression()
#rfe = RFE(model, 3)
#fit = rfe.fit(X, Y)
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
#print("Feature Ranking: %s") % fit.ranking_



validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# Spot-check algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

# Compare algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))


results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

# Compare algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()




# ~~~~~~~~~~~~~~5. Improve Accuracy~~~~~~~~~~~~~~~~~
# a) Algorithm Tuning
# b) Ensembles

# CART Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
param_grid = dict(max_depth = k_values)
model = DecisionTreeRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param))





# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use
