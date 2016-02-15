import pandas
import numpy
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing

wineData = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

y = wineData[0]
X = wineData.loc[:, 1:13]

kfoldGen = KFold(n=y.shape[0], n_folds=5, shuffle=True, random_state=42)

res = pandas.DataFrame(columns=['k', 'score'])

for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k)
    sc = cross_validation.cross_val_score(clf, X, y, cv=kfoldGen)
    msc = numpy.mean(sc)
    res.loc[k] = [k, msc]

print('Original:')
print(res.ix[res['score'].idxmax()].to_string())

X = preprocessing.scale(X)

res = pandas.DataFrame(columns=['k', 'score'])

for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k)
    sc = cross_validation.cross_val_score(clf, X, y, cv=kfoldGen)
    msc = numpy.mean(sc)
    res.loc[k] = [k, msc]

print('\nScaled:')
print(res.ix[res['score'].idxmax()].to_string())
