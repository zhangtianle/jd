import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb

from tl.src.util import read_data


def delete(x, test, coloumn):
    return x.pop(coloumn), test.pop(coloumn)


def xgb_train(X, Y):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth': 8, 'eta': 0.1, 'silent': 1, 'eval_metric': 'rmse'}
    # alternatively:
    plst = param.items()
    # plst += [('eval_metric', 'ams@0')]

    num_round = 40
    bst = xgb.train(plst, dtrain, num_round, evallist)
    # make prediction
    preds = bst.predict(dtest)

    xgb.plot_importance(bst)
    xgb.plot_tree(bst, num_trees=2)

def xgb_train_online(X, Y, Test, uid):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    test_X = Test.as_matrix()

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X)
    evallist = [(dtrain, 'train')]

    param = {'max_depth': 8, 'eta': 0.1, 'silent': 1, 'eval_metric': 'rmse'}
    # alternatively:
    plst = param.items()
    # plst += [('eval_metric', 'ams@0')]

    num_round = 40
    bst = xgb.train(plst, dtrain, num_round, evallist)
    # make prediction
    predict = bst.predict(dtest)

    xgb.plot_importance(bst)
    xgb.plot_tree(bst, num_trees=2)

    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    result = pd.DataFrame()
    result[0] = uid
    result[1] = predict
    result.to_csv("../result/result_11.22_xbg.csv", header=None, index=False, encoding="utf-8")


def offline(X, Y):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    train_X = data_scaler(train_X)

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                    n_estimators=500,
                                    learning_rate=0.05,
                                    max_depth=10,
                                    subsample=0.8,
                                    min_samples_split=9,
                                    max_leaf_nodes=10)
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_train)
    print(clf.score(X_train, y_train))
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    print("GBDT: Mean squared train error: %.2f" % mean_squared_error(y_train, predict))

    predict = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    print("GBDT: Mean squared test error: %.2f" % mean_squared_error(y_test, predict))

    predictors = [x for x in X.columns]
    feat_imp = pd.Series(clf.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_train)
    print(clf.score(X_test, y_test))
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    print("Linear Regression: Mean squared train error: %.2f" % mean_squared_error(y_train, predict))

    predict = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    print("Linear Regression: Mean squared test error: %.2f" % mean_squared_error(y_test, predict))


def online_LR(X, Y, Test, uid):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()
    test_X = Test.as_matrix()

    clf = LinearRegression()
    clf.fit(train_X, train_Y)
    predict = clf.predict(test_X)
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0

    result = pd.DataFrame()
    result[0] = uid
    result[1] = predict
    result.to_csv("../result/result_11.20_LR.csv", header=None, index=False, encoding="utf-8")


def online_GBDT(X, Y, Test, uid):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()
    test_X = Test.as_matrix()

    train_X = data_scaler(train_X)
    test_X = data_scaler(test_X)

    clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                    n_estimators=500,
                                    learning_rate=.02,
                                    max_depth=8,
                                    subsample=0.8, min_samples_leaf=9,
                                    min_samples_split=9,
                                    max_leaf_nodes=10)
    clf.fit(train_X, train_Y)
    predict = clf.predict(test_X)
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    result = pd.DataFrame()
    result[0] = uid
    result[1] = predict
    result.to_csv("../result/result_11.21_GBDT.csv", header=None, index=False, encoding="utf-8")


def data_scaler(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data)


def main():
    loan, user, order, click = read_data()
    uid = pd.DataFrame(user["uid"])

    X = pd.DataFrame(pd.read_csv("../feature/train_x_offline.csv"))
    Y = pd.DataFrame(pd.read_csv("../feature/train_y_offline.csv"))

    Test = pd.DataFrame(pd.read_csv("../feature/test_x_online.csv"))

    X = X.fillna(0)
    Test = Test.fillna(0)

    _, uid = delete(X, Test, "uid")
    # delete(X, Test, "average_discount")

    Y.pop("uid")

    # offline(X, Y)
    # xgb_train(X, Y)
    xgb_train_online(X, Y, Test, uid)
    # online_GBDT(X, Y, Test, uid)
    # online_LR(X, Y, Test, uid)


if __name__ == "__main__":
    main()
