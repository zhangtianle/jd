import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd

from tl.src.util import read_data


class MyModel:
    def __init__(self):
        self.xgb_r_param = {'max_depth': 5, 'eta': 0.05, 'silent': 1, 'eval_metric': 'rmse', 'max_leaf_nodes': 4}
        self.xgb_c_param = {'objective': 'binary:logistic', 'max_depth': 6, 'eta': 0.05, 'silent': 1,
                            'eval_metric': 'error', 'max_leaf_nodes': 10}

        self.xgb_r_num_round = 300
        self.xgb_c_num_round = 80

        self.param_grid = ParameterGrid({'max_depth': [6, 7, 8, 9, 10, 11, 12],
                                         'eta': [0.02, 0.05, 0.1, 0.2],
                                         'silent': 1,
                                         'eval_metric': 'rmse',
                                         'max_leaf_nodes': [10, 15, 20, 25, 30]
                                         })

    def main(self, data):
        # GridSearchCV(estimator=XGBRegressor()
        xgb.grid(self.param_grid, data, self.xgb_r_num_round, 5)


if __name__ == '__main__':
    my_model = MyModel()
    loan, user, order, click = read_data()
    uid = pd.DataFrame(user["uid"])

    X = pd.DataFrame(pd.read_csv("../feature/train_x_offline.csv"))
    Y = pd.DataFrame(pd.read_csv("../feature/train_y_offline.csv"))

    Test = pd.DataFrame(pd.read_csv("../feature/test_x_online.csv"))

    X = X.fillna(0)
    Test = Test.fillna(0)
    # my_model.main(data)