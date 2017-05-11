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


# define base model
def baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  
  # Compile model
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
