import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn import cross_validation

abalone = pd.read_csv("week5-q1\\abalone.csv")

abalone['Sex'] = abalone['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = abalone.iloc[:, 0:8]
y = abalone.iloc[:, 8]

kfoldGen = KFold(n=y.shape[0], n_folds=5, shuffle=True, random_state=1)

res = pd.DataFrame(columns=['k', 'score'])

for k in range(1, 51):
    clf = RandomForestRegressor(n_estimators=k, random_state=1)
    sc = cross_validation.cross_val_score(clf, X, y, cv=kfoldGen, scoring='r2')
    msc = np.mean(sc)
    res.loc[k] = [k, msc]

with open('week5-q1\\1.txt', 'w') as f1:
    f1.write(str(int(res.query('score > 0.52')['k'].min())))
