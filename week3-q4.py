import numpy as np
import pandas
import math
from sklearn import metrics

# читаем данные
clsdata = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\week3-q4\\classification.csv")

TP = np.sum((clsdata['true'] == 1) & (clsdata['pred'] == 1))
FP = np.sum((clsdata['true'] == 0) & (clsdata['pred'] == 1))

FN = np.sum((clsdata['true'] == 1) & (clsdata['pred'] == 0))
TN = np.sum((clsdata['true'] == 0) & (clsdata['pred'] == 0))

print("TP FP FN TN: ", " ".join(map(str, [TP, FP, FN, TN])))

acc = metrics.accuracy_score(clsdata['true'], clsdata['pred'])
pre = metrics.precision_score(clsdata['true'], clsdata['pred'])
rec = metrics.recall_score(clsdata['true'], clsdata['pred'])
f1 = metrics.f1_score(clsdata['true'], clsdata['pred'])

print("acc pre rec f1: ", " ".join(map(str, np.round([acc, pre, rec, f1], 2))))

# -------
sc = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\week3-q4\\scores.csv")

ar = {
    'score_logreg': metrics.roc_auc_score(sc['true'], sc['score_logreg']),
    'score_svm': metrics.roc_auc_score(sc['true'], sc['score_svm']),
    'score_knn': metrics.roc_auc_score(sc['true'], sc['score_knn']),
    'score_tree': metrics.roc_auc_score(sc['true'], sc['score_tree'])}

print("Best method: ", max(ar, key=ar.get))


def MaxPrecisionWithRecall70(y_true, probas_pred):
    rc = metrics.precision_recall_curve(y_true, probas_pred)
    p = pandas.DataFrame({'precision': rc[0], 'recall': rc[1]})
    p70 = p[p['recall'] >= .70]
    return np.max(p70['precision'])


maxp70 = {
    'score_logreg': MaxPrecisionWithRecall70(sc['true'], sc['score_logreg']),
    'score_svm': MaxPrecisionWithRecall70(sc['true'], sc['score_svm']),
    'score_knn': MaxPrecisionWithRecall70(sc['true'], sc['score_knn']),
    'score_tree': MaxPrecisionWithRecall70(sc['true'], sc['score_tree'])}

print("Max precision: ", round(maxp70[max(maxp70)],2), " with ", max(maxp70))
