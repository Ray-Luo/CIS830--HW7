# Rescale data (between 0 and 1)
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

# load dataset
def scale_standardize(X):
#filename = filename = 'dataset_2_CPU.csv'
#dataset = read_csv(filename, header = 0)

#array = dataset.values

#X = array[:,0:8]
#Y = array[:,8]

# examine if the index is correct
#print(X[0:5,:])
#print('******************')
#print(Y[0:5])
#print('******************')
  a = list()
  
  []

  scaler = MinMaxScaler(feature_range=(0, 1))
  rescaledX = scaler.fit_transform(X)

# summarize transformed data
#print(rescaledX[0:5,:])

# standardize scaled data
  scaler = StandardScaler().fit(rescaledX)
  standardized_rescaledX = scaler.transform(rescaledX)

#
  scaler = StandardScaler().fit(standardized_rescaledX)
  normalizedX = scaler.transform(standardized_rescaledX)

# summarize standized data
  return normalizedX
