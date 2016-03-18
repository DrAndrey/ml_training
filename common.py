# -*- coding: utf-8 -*-

"""

"""

import os

import pylab
import numpy as np
import pandas as pd


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

if __name__ == '__main__':
    pass
