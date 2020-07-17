import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Uni_Functions.functs as fu
np.set_printoptions(suppress=True)

data = pd.read_csv('Dataset/dataset.csv')

X = data["Head Size(cm^3)"].values
y = data["Brain Weight(grams)"].values

# Editing our feature X and Label Y
X = np.array(X, dtype='float64')
y = np.array(y, dtype='float64')
y = np.reshape(y, (len(y), 1))

# Adding bias column to our input feature
X = np.column_stack([np.ones(len(X)), X])

# Initialization of our theta (2x1 vector)
theta = np.zeros((2, 1), dtype='float64')

# Optimized value for theta based on the Normal equation method
theta_trix = fu.matrix_imp(X, y)

x_new_trix = 5034

# Add ones column to our new x instance
if np.isscalar(x_new_trix):
    x_new_trix = np.array([1, x_new_trix], dtype='float64')
else:
    x_new_trix = np.column_stack(
        [np.ones(len(x_new_trix), dtype='float64'), x_new_trix])

# Predicted Value for y based on our new instance
predicted_value = fu.prediction(x_new_trix, theta_trix)

print(theta_trix)
cost = fu.cost_comp(X, y, theta_trix)
