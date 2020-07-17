import pandas as pd
from statsmodels.regression.linear_model import OLS
import numpy as np
np.set_printoptions(suppress=True)

data = pd.read_csv('Dataset/dataset.csv')


X = data["Head Size(cm^3)"].values
y = data["Brain Weight(grams)"].values

X = np.array(X, dtype='float64')
y = np.array(y, dtype='float64')
y = np.reshape(y, (len(y), 1))

X = np.column_stack([np.ones(len(X)), X])

# Implement the statsmodel function

res = OLS(y, X).fit()

# Theta values
theta = res.params

print(theta)

# prediction
ols_pred = res.predict()

print(res.summary())
