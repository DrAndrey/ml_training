# -*- coding: utf-8 -*-

"""

"""

import os
import collections
import multiprocessing as mp

import pylab
import numpy as np
import pandas as pd
import hyperopt

from sklearn.metrics import get_scorer
from sklearn.cross_validation import StratifiedKFold, cross_val_score

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


def get_most_important_features(estimator, columns):
    most_important_features = collections.OrderedDict()
    scores = reversed(sorted(estimator._Booster.get_fscore().items(), key=lambda x: x[1]))
    for feature_key, score in scores:
        most_important_features[columns[int(feature_key[1:])]] = score
    return most_important_features


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


class HyperoptTester:

    def __init__(self, estimator_factory, opt_space, random_state, nf_test=5, nf_val=5):
        self.estimator_factory = estimator_factory
        self.opt_space = opt_space
        self.random_state = random_state
        self.nf_test = nf_test
        self.nf_val = nf_val

        self.counter = 0
        self.opt_params = {k: [0, {}] for k in range(self.nf_test)}

    def optimize(self, x, y, scoring, max_evals=100):
        scorer = get_scorer(scoring)
        test_cv = StratifiedKFold(y, n_folds=self.nf_test, shuffle=True, random_state=self.random_state)
        for num_test_step, indexes in enumerate(test_cv):
            val_index, test_index = indexes
            val_x = x[val_index]
            val_y = y[val_index]

            trials = hyperopt.Trials()
            self.counter = 0
            opt_fun = self.make_opt_fun(val_x, val_y, scoring, num_test_step)
            hyperopt.fmin(opt_fun, self.opt_space, algo=hyperopt.tpe.suggest, max_evals=max_evals, trials=trials)

        scores = []
        for num_cv_step, indexes in enumerate(test_cv):
            val_index, test_index = indexes
            val_x, test_x = x[val_index], x[test_index]
            val_y, test_y = y[val_index], y[test_index]

            cv_part_scores = []
            for num_test_step in range(self.nf_test):
                print("num_cv_step - {0}, num_test_step - {1}".format(num_cv_step, num_test_step))
                estimator = self.estimator_factory(**self.opt_params[num_test_step][1])
                estimator.fit(val_x, val_y)
                score = scorer(estimator, test_x, test_y)
                cv_part_scores.append(score)
            scores.append(cv_part_scores)
        return np.array(scores)

    def make_opt_fun(self, val_x, val_y, scoring, num_test_step):
        val_cv = StratifiedKFold(val_y, n_folds=self.nf_val, shuffle=True, random_state=self.random_state)

        def hyperopt_train_test(params):
            estimator = self.estimator_factory(**params)
            return cross_val_score(estimator, val_x, val_y, scoring=scoring, cv=val_cv, n_jobs=NCPU, verbose=1).mean()

        def opt_fun(params):
            acc = hyperopt_train_test(params)
            if acc > self.opt_params[num_test_step][0]:
                self.opt_params[num_test_step][0] = acc
                self.opt_params[num_test_step][1] = params
                print("new best score - {0}, best params - {1}, num test step - {2}".format(acc, params, num_test_step))

            print("iters - {0}, num test step - {1}".format(self.counter, num_test_step))
            self.counter += 1
            return {"loss": -acc, "status": hyperopt.STATUS_OK}

        return opt_fun

if __name__ == '__main__':
    import xgboost


