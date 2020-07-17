import pandas as pd
import numpy as np

data = pd.read_csv('Dataset/dataset.csv')

X = data["Head Size(cm^3)"].values
y = data["Brain Weight(grams)"].values


def mean(param):
    return sum(param)/float(len(param))


def variance(param):
    val_mean = mean(param)
    return sum([(x-val_mean)**2 for x in param])


def covariance(x, x_mean, y, y_mean):
    covr = 0.0
    for i in range(len(x)):
        covr += (x[i]-x_mean) * (y[i]-y_mean)

    return covr


def get_Thetas(x, y):
    x_mean = mean(x)
    y_mean = mean(y)
    covr = covariance(x, x_mean, y, y_mean)

    x_var = variance(x)
    theta = [0 for zeros in range(2)]
    theta[0] = covr/x_var
    theta[1] = np.mean(y) - theta[0] * np.mean(x)

    return theta


def prediction(x, y):
    theta = get_Thetas(x, y)
    predi = theta[0]+(theta[1]*x)
    return predi


prediction = prediction(X, y)

print(prediction)
