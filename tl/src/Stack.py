from sys import path

path.append('../../')
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tl.src.util import delete, error, save_to_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlxtend.regressor import StackingRegressor
import matplotlib.pyplot as plt


class Stack:
    def __init__(self):
        self.parameter_init()

    def parameter_init(self):
        self.X = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_8_end_10.csv"))
        self.Y = pd.DataFrame(pd.read_csv("../feature/train_y_11_offline.csv"))

        self.Test = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_8_end_11.csv"))

        self.X = self.X.fillna(0)
        self.Test = self.Test.fillna(0)

        _, self.uid = delete(self.X, self.Test, "uid")
        # delete(X, Test, "average_discount")

        self.Y.pop("uid")

    def stackModel(self):
        train_X = self.X.as_matrix()
        train_Y = self.Y.as_matrix()

        test_X = self.Test.as_matrix()

        # train_X = data_scaler(train_X)

        X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

        gbdt = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                         n_estimators=500,
                                         learning_rate=0.05,
                                         max_depth=8,
                                         subsample=0.8,
                                         min_samples_split=9,
                                         max_leaf_nodes=10)
        xgb = XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.05, silent=False)
        lr = LinearRegression()
        rfg = RandomForestRegressor(bootstrap=False, max_features=0.05, min_samples_leaf=11, min_samples_split=8,
                                    n_estimators=100)
        svr_rbf = SVR(kernel='rbf')

        stregr = StackingRegressor(regressors=[gbdt, xgb, lr, rfg], meta_regressor=svr_rbf)

        stregr.fit(X_train, y_train)
        stregr.predict(X_train)

        # Evaluate and visualize the fit

        print("Mean Squared Error: %.6f" % np.mean((stregr.predict(X_train) - y_train) ** 2) ** 0.5)
        error(stregr.predict(X_test), y_test)

        # online
        result = stregr.predict(test_X)
        save_to_file(result, self.uid, "../result/result_12.09_2_stacking.csv")

        with plt.style.context(('seaborn-whitegrid')):
            plt.scatter(X_train, y_train, c='lightgray')
            plt.plot(X_train, stregr.predict(X_train), c='darkgreen', lw=2)

        plt.show()


if __name__ == '__main__':
    stack = Stack()
    stack.stackModel()
