# -*- coding: utf-8 -*-

"""

"""

import os
import multiprocessing as mp

import pylab
import numpy as np
import pandas as pd

from sklearn.metrics import get_scorer

NCPU = mp.cpu_count() - 2


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


def _calc_score(kwargs):
    cv_num = kwargs["cv_num"]+1
    print("{0} fold is running".format(cv_num))
    kwargs["estimator"].fit(kwargs["x_train"], kwargs["y_train"], kwargs["w"])
    score = kwargs["scorer"](kwargs["estimator"], kwargs["x_test"], kwargs["y_test"])
    print("{0} fold score - {1}".format(cv_num, score))
    return score


def cross_val_score_with_weights(estimator, x, y, w, scoring, cv):
    scorer = get_scorer(scoring)

    args = []
    for cv_num, inds in enumerate(cv):
        train_ind, test_ind = inds
        arg_set = {"cv_num": cv_num, "x_train": x[train_ind], "x_test": x[test_ind], "y_train": y[train_ind],
                   "y_test": y[test_ind], "w": w[train_ind], "estimator": estimator, "scorer": scorer}
        args.append(arg_set)

    with mp.Pool(NCPU) as pool:
        scores = pool.map(_calc_score, args)
    return np.array(scores)


if __name__ == '__main__':
    import xgboost
    import hyperopt

