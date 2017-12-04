from sys import path
path.append('../../')
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb

from tl.src.MyModel import MyModel


def delete(x, test, coloumn):
    return x.pop(coloumn), test.pop(coloumn)


def error(y_train, predict):
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    print("Mean squared train error: %.6f" % mean_squared_error(y_train, predict) ** 0.5)


def xgb_classify(X, Y):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    for i in range(len(train_Y)):
        if train_Y[i] != 0:
            train_Y[i] = 1
        else:
            train_Y[i] = 0

    train_X = data_scaler(train_X)

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    model = MyModel()
    bst = xgb.train(model.xgb_c_param, dtrain, model.xgb_c_num_round, evallist)
    return bst.predict(dtest)


def xgb_classify_online(X, Y, Test, uid):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()
    test_X = Test.as_matrix()

    for i in range(len(train_Y)):
        if train_Y[i] != 0:
            train_Y[i] = 1
        else:
            train_Y[i] = 0

    train_X = data_scaler(train_X)

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X)
    evallist = [(dtrain, 'train')]

    dtrain = xgb.DMatrix(train_X, label=train_Y)

    model = MyModel()

    bst = xgb.train(model.xgb_c_param, dtrain, model.xgb_c_num_round, evallist)
    # make prediction
    predict = bst.predict(dtest)
    return predict


def xgb_train(X, Y):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    train_X = data_scaler(train_X)

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    model = MyModel()

    # num_round = 150
    bst = xgb.train(model.xgb_r_param, dtrain, model.xgb_r_num_round, evallist)
    # bst = xgb.cv(model.xgb_r_param, dtrain, model.xgb_r_num_round, nfold=5, metrics={'error'}, seed=0, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

    # make prediction
    preds = bst.predict(dtest)
    xgb.plot_importance(bst)
    # xgb.plot_tree(bst, num_trees=2)
    plt.show()

    # 加上分类的结果
    # classify = xgb_classify(X, Y)
    # for i in range(len(preds)):
    #     if classify[i] == 0:
    #         preds[i] = 0
    # error(y_test, preds)

    return preds


def xgb_train_online(X, Y, Test, uid):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    test_X = Test.as_matrix()

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X)
    evallist = [(dtrain, 'train')]

    model = MyModel()

    bst = xgb.train(model.xgb_r_param, dtrain, model.xgb_r_num_round, evallist)
    # make prediction
    predict = bst.predict(dtest)

    # xgb.plot_importance(bst)
    # xgb.plot_tree(bst, num_trees=2)

    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    result = pd.DataFrame()
    result[0] = uid
    result[1] = predict
    result.to_csv("../result/result_11.29_1_xbg.csv", header=None, index=False, encoding="utf-8")


def offline(X, Y):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    # train_X = data_scaler(train_X)

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                    n_estimators=500,
                                    learning_rate=0.05,
                                    max_depth=8,
                                    subsample=0.8,
                                    min_samples_split=9,
                                    max_leaf_nodes=10)
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_train)
    print(clf.score(X_train, y_train))
    error(y_train, predict)

    predict = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    error(y_test, predict)

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

    # train_X = data_scaler(train_X)
    # test_X = data_scaler(test_X)

    clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                    n_estimators=500,
                                    learning_rate=.05,
                                    max_depth=8,
                                    subsample=0.8,
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
    result.to_csv("../result/result_12.04_1_GBDT.csv", header=None, index=False, encoding="utf-8")


def data_scaler(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data)


def main():

    # X = pd.DataFrame(pd.read_csv("../feature/train_x_offline.csv"))
    # Y = pd.DataFrame(pd.read_csv("../feature/train_y_offline.csv"))
    #
    # Test = pd.DataFrame(pd.read_csv("../feature/test_x_online.csv"))

    X = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_8_end_10.csv"))
    Y = pd.DataFrame(pd.read_csv("../feature/train_y_11_offline.csv"))

    Test = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_9_end_11.csv"))

    X = X.fillna(0)
    Test = Test.fillna(0)

    _, uid = delete(X, Test, "uid")
    # delete(X, Test, "average_discount")

    Y.pop("uid")

    # xgb参数
    model = MyModel()

    # classify = xgb_classify(X, Y)

    # offline(X, Y)
    # preds = xgb_train(X, Y)
    # xgb_train_online(X, Y, Test, uid)
    online_GBDT(X, Y, Test, uid)
    # online_LR(X, Y, Test, uid)


if __name__ == "__main__":
    main()
