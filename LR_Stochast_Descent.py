# Import Libraries
from random import randint
'exec(%matplotlib inline)'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions.costFunction as cfx
plt.rcParams['figure.figsize'] = (12.0, 9.0)


df = pd.read_csv('Datasets/dataset.csv')
X = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values
np.set_printoptions(suppress=True)

m = len(X)
ones = np.ones((m, 1))

# Initialize our X and y
y = np.array(y, dtype='float64')
y = np.reshape(y, (m, 1))
X = np.array(X, dtype='float64')
X = np.reshape(X, (m, 1))

# Adding ones column to X
X = np.append(ones, X, axis=1)

# Initialise our θ parameter

theta = np.zeros((2, 1), dtype='float64')

# Calculating cost
cost = cfx.J_cost(X, y, theta)

# Stochastic Gradient Descent
theta, J_history = cfx.gradient_descent_stochast(X, y, theta)


"""
    Based on our optimal values for Theta lets plot predictions
"""
new_X_instance = np.array([[randint(1000, 10000)], [randint(1000, 10000)]])
print(new_X_instance)

# Add new instance to X array
X_new = np.c_[np.ones((2, 1)), new_X_instance]
y_predict = X_new.dot(theta)

"""
plt.plot(X_new[:,1:], y_predict, 'r-')
plt.plot(X[:, 1:],y,'b.')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')

plt.show()
"""

"""
    Graphing our descent (otherwise known as J history)


fig,ax= plt.subplots(figsize=(12,8))

ax.set_xlabel('Iterations')
ax.set_ylabel('J_theta')

_ = ax.plot(range(1500), J_history[:1500], 'b.')
plt.show()
"""
