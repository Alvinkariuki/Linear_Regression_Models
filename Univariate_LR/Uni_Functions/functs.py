import numpy as np

#  Matrix Form Implementation


def matrix_imp(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return theta


# Function to predict linear regression using normal equation

def prediction(X_new, theta):
    pred_y = np.dot(X_new, theta)

    return pred_y


def cost_comp(X, y, theta):
    m = len(X)
    h = X.dot(theta)
    sqrd_errs = np.sum(np.square(h-y))
    J = (sqrd_errs)/(2*m)
    return J
