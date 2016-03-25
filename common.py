# -*- coding: utf-8 -*-

"""

"""

import os
import multiprocessing

import pylab
import numpy as np
import pandas as pd

from sklearn.metrics import get_scorer

NCPU = multiprocessing.cpu_count() - 1


def save_output(data, labels):
    df = pd.DataFrame(data, columns=labels)
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df.to_csv(os.path.join("output", "output.csv"), index=False)


def get_independent_features_mask(matr, tol=1e-5, is_plot=False):
    r = np.linalg.qr(matr)[1]
    r_sum = abs(r).sum(axis=1)
    r_sum /= np.linalg.norm(r_sum)

    mask = r_sum > tol
    if is_plot:
        pylab.plot(sorted(np.log(r_sum[mask])))
        pylab.show()
    return mask


def find_corr_features_mask(matr, trashhold=0.9):
    matr_cc = np.corrcoef(matr, rowvar=0)
    corr_ind = []
    n = matr_cc.shape[0]
    for i in range(n-1):
        for j in range(i+1, n):
            if matr_cc[i, j] > trashhold:
                corr_ind.append(j)
    corr_mask = np.array([False if i in corr_ind else True for i in range(n)])
    return corr_mask


def cross_val_score_with_weights(estimator_cls, x, y, w, scoring, cv, params):
    scores = []
    scorer = get_scorer(scoring)
    for train_ind, test_ind in cv:
        estimator = estimator_cls(**params)
        estimator.fit(x[train_ind], y[train_ind], w[train_ind])

        if scoring in ["roc_auc"]:
            y_pred = estimator.predict_proba(x[test_ind])[:, 1]
        else:
            y_pred = estimator.predict(x[test_ind])[:, 1]
        scores.append(scorer(y[test_ind], y_pred))
    return np.array(scorer)


if __name__ == '__main__':
    pass
