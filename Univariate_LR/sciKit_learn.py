from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

data = pd.read_csv('Dataset/dataset.csv')

X = data["Head Size(cm^3)"].values
y = data["Brain Weight(grams)"].values

X = np.array(X, dtype='float64')
y = np.array(y, dtype='float64')
y = np.reshape(y, (len(y), 1))

X = np.column_stack([np.ones(len(X)), X])


lr = LinearRegression()
model = lr.fit(X, y)

# Model Accuracy
acc = model.score(X, y)

# Model Prediction
preds_y = model.predict(X)

x_new = 4500

if np.isscalar(x_new):
    x_new = np.array([1, x_new])
else:
    x_new = np.column_stack([np.ones(len(x_new), dtype='float64'), x_new])

pred_new = model.predict(x_new.reshape(1, -1))  # Pred for new x

# Model Evaluation   (explained_variance  == r2)
explained_variance = explained_variance_score(y, preds_y)

r_sqrd = r2_score(y, preds_y)

mean_abs_err = mean_absolute_error(y, preds_y)

mean_sqrd_err = mean_squared_error(y, preds_y)
