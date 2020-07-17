import numpy as np


def J_cost(X, y, theta):
    m = len(X)
    h = X.dot(theta)
    sqrd_errs = np.sum(np.square(h-y))
    J = (sqrd_errs)/(2*m)
    return J


# Classic Gradient Descent
def gradient_descent(X, y, theta, alpha=0.00000001, iterations=1500):
    m = len(y)
    J_hist = np.zeros(iterations)
    for iter in range(iterations):
        h = X.dot(theta)
        theta = theta - (1/m)*alpha*(X.T.dot((h-y)))
        J_hist[iter] = J_cost(X, y, theta)

    return theta, J_hist

# Stochastic Gradient Descent


def gradient_descent_stochast(X, y, theta, alpha=0.00000001, iterations=1500):
    m = len(y)
    J_hist = np.zeros(iterations)

    for iter in range(iterations):
        rand_indx = np.random.randint(0, m)
        x_rand = X[rand_indx, :].reshape(1, X.shape[1])
        y_rand = y[rand_indx].reshape(1, 1)
        h = np.dot(x_rand, theta)

        theta = theta - (1/m)*alpha*(x_rand.T.dot(h-y_rand))
        J_hist[iter] = J_cost(x_rand, y_rand, theta)

    return theta, J_hist

# Mini batch Gradient Descent


def gradient_descent_miniB(X, y, theta, alpha=0.00000001, iterations=1500, batch_size=20):
    m = len(y)
    J_hist = np.zeros(iterations)

    for iter in range(iterations):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]

        for i in range(0, m, batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            h = np.dot(x_batch, theta)

            theta = theta - (1/m)*alpha*(x_batch.T.dot((h-y_batch)))
            J_hist[iter] = J_cost(x_batch, y_batch, theta)

    return theta, J_hist
