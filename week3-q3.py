import numpy as np
import pandas
import math

from sklearn import metrics

# читаем данные
dl = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\week3-q3\\data-logistic.csv", header=None)

# разделяем на признаки и результат
y = dl[0]
X = dl.loc[:, 1:2]

def grads(w1, w2, X, y, C, k):
    eps = 1e-5
    l = y.shape[0]

    for i in range(10000):
        nw1 = w1 + k / l * np.sum(X[1] * y * (1 - 1 / (1 + np.exp(-y * (w1 * X[1] + w2 * X[2]))))) - k * C * w1
        nw2 = w2 + k / l * np.sum(X[2] * y * (1 - 1 / (1 + np.exp(-y * (w1 * X[1] + w2 * X[2]))))) - k * C * w2

        dist = math.sqrt((nw1 - w1) ** 2 + (nw2 - w2) ** 2)
        if dist < eps:
            break

        w1 = nw1
        w2 = nw2

    res = [w1, w2]
    print("Шагов: ", i)
    print(res)

    #y_vals = np.sign(w1 * X[1] + w2 * X[2])
    y_scores = 1 / (1 + np.exp(-w1 * X[1] - w2 * X[2]))
    s = metrics.roc_auc_score(y, y_scores)

    print("Score: ", s)

    return [w1, w2]

w_C00 = grads(0., 0., X, y, 0., 0.1)
w_C10 = grads(0., 0., X, y, 10., 0.1)


