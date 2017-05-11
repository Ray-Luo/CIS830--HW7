import numpy
from pandas import DataFrame
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from transform import scale_standardize
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from IPython.display import Image
import pydotplus

# load dataset
filename = "new_dataset_Mem.csv"
dataset = read_csv(filename)


# split into input and output variables
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


X = Z[:,0:7]
Y = Z[:,7]


# define GP model
model = SymbolicRegressor(population_size=10000, generations=20, tournament_size=20, stopping_criteria=130.0, const_range=(-1.0, 1.0), init_depth=(2, 6), init_method='half and half', function_set=('add', 'sub', 'mul', 'div','log','inv','sin','cos','tan'), metric='mse', parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0, warm_start=False, n_jobs=1, verbose=0, random_state=None)




# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#evaluate model with standardized dataset
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)))
#pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)

mse = list()

for train_index, test_index in kfold.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = Y[train_index], Y[test_index]
  model.fit(X_train, y_train)
  y_output = model.predict(X_test)
  for i in range(len(X_test)):
    mse.append((y_test[i] - y_output[i])**2)

mse = numpy.asarray(mse)
print("Mean: %.2f (%.2f)" % (mse.mean(), mse.std()))

