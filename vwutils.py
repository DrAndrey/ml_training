# -*- coding: utf-8 -*-

"""

"""

import numpy as np
import pandas as pd


def make_vw_input(x, input_file, y=None, is_log_hinge_loss=True):
    feature_names = x.columns.values

    if y is None:
        y = pd.Series(np.ones(len(x), dtype=np.int))

    if is_log_hinge_loss:
        y = y.copy()
        y.loc[y == 0] = -1

    column_names = {key: val for val, key in enumerate(feature_names)}
    with open(input_file, "w") as f_out:
        for y_val, row in zip(y.values, x.iterrows()):
            feature_part = " ".join(["{0}:{1}".format(column_names[feature_name], row[1][feature_name])
                                     for feature_name in feature_names])
            w = 1
            string_row = "{0} {1} |f {2}\n".format(y_val, w, feature_part)

            f_out.write(string_row)

if __name__ == '__main__':
    pass