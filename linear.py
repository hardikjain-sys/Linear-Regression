
import numpy as np
import pandas as pd

url ="https://raw.githubusercontent.com/hardikjain-sys/Linear-Regression/refs/heads/main/housing.csv"


data = pd.read_csv(url)

data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())

X = data.drop('median_house_value', axis=1).copy()

nColumns = X.select_dtypes(include=["float64", "int64"]).columns
X = pd.get_dummies(X, columns=['ocean_proximity'])
X = X.astype(int)
weights = np.ones((X.shape[1],1),dtype="float64")
bias = 0.0


for col in nColumns:
    MIN = X[col].min()
    MAX = X[col].max()
    d = MAX - MIN
    X[col] = (X[col] - MIN)/float(d)

Y = data['median_house_value'].to_numpy().reshape(-1, 1)
X = X.to_numpy()

def gradientDescent(learningRate):
    global weights, bias
    yCap = (X @ weights) + bias
    errorV = yCap - Y

    diff = np.zeros(X.shape[1], dtype="float64")
    for i in range(X.shape[1]):
        diff[i] = np.sum(errorV.flatten() * X[:, i]) * 2 / X.shape[0]

    biasDiff = np.sum(errorV) * 2 / X.shape[0]

    newWeights = np.zeros((X.shape[1], 1), dtype="float64")
    for i in range(X.shape[1]):
        newWeights[i][0] = weights[i][0] - learningRate * diff[i]

    biasNew = bias - learningRate * biasDiff

    return newWeights, biasNew


learningRate = 0.2
itr = 10000

for i in range(itr):
    weights,bias =gradientDescent(learningRate)


def checkModel():
    Y_pred = (X @ weights) + bias
    Y_mean = np.mean(Y)
    ssr = np.sum(np.square(Y - Y_pred))
    sst = np.sum(np.square(Y - Y_mean))
    if sst != 0:
        R_squared = 1 - (ssr / sst)
    else:
        R_squared = 0.0

    print(R_squared)

checkModel()
