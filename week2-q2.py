import numpy
import pandas
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import preprocessing

boston = load_boston()

X = preprocessing.scale(boston.data)
y = boston.target

kfoldGen = KFold(n=y.shape[0], n_folds=5, shuffle=True, random_state=42)

res = pandas.DataFrame(columns=['p', 'score'])

for p in numpy.linspace(1, 10, num=200):
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    sc = cross_validation.cross_val_score(clf, X, y, cv=kfoldGen, scoring='mean_squared_error')
    msc = numpy.mean(sc)
    res.loc[res.shape[0]] = [p, msc]

print(res.ix[res['score'].idxmax()].to_string())
