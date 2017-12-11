from sys import path

from sklearn.feature_selection import SelectFwe, f_regression

path.append('../../')
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tpot.builtins import StackingEstimator
from tl.src.util import error, delete, save_to_file
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoLarsCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

from tl.src.MyModel import MyModel


def ptop_2030(X, Y, Test, uid, online=0):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    test_X = Test.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    # Score on the training set was:-3.207903288331976
    exported_pipeline = make_pipeline(
        SelectFwe(score_func=f_regression, alpha=0.038),
        StackingEstimator(estimator=LassoLarsCV(normalize=False)),
        StackingEstimator(
            estimator=GradientBoostingRegressor(alpha=0.9, learning_rate=1.0, loss="quantile", max_depth=4,
                                                max_features=0.8, min_samples_leaf=7, min_samples_split=17,
                                                n_estimators=100, subsample=0.1)),
        ExtraTreesRegressor(bootstrap=True, max_features=0.8500000000000001, min_samples_leaf=12, min_samples_split=10,
                            n_estimators=100)
    )

    exported_pipeline.fit(X_train, y_train)
    print("train:--------------")
    predict = exported_pipeline.predict(X_train)
    error(predict, y_train)

    print("test:---------------")
    predict = exported_pipeline.predict(X_test)
    error(predict, y_test)

    # online
    if online == 1:
        exported_pipeline.fit(train_X, train_Y)
        predict = exported_pipeline.predict(test_X)
        save_to_file(predict, uid, "../result/result_12.11_1_ptot2030.csv")


def ptop_1050(X, Y, Test, uid, online=0):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    test_X = Test.as_matrix()

    # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    # Score on the training set was:-3.196408119379142
    exported_pipeline = make_pipeline(
        MinMaxScaler(),
        GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="ls", max_depth=5, max_features=0.2,
                                  min_samples_leaf=3, min_samples_split=20, n_estimators=100,
                                  subsample=0.9500000000000001)
    )

    # exported_pipeline.fit(X_train, y_train)
    # print("train:--------------")
    # predict = exported_pipeline.predict(X_train)
    # error(predict, y_train)
    #
    # print("test:---------------")
    # predict = exported_pipeline.predict(X_test)
    # error(predict, y_test)

    # online
    if online == 1:
        exported_pipeline.fit(train_X, train_Y)
        predict = exported_pipeline.predict(test_X)
        save_to_file(predict, uid, "../result/result_12.11_1_ptop_10503.csv")


def ptop_2040(X, Y, Test, uid, online=0):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    test_X = Test.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=LassoLarsCV(normalize=False)),
        RobustScaler(),
        GradientBoostingRegressor(alpha=0.8, learning_rate=0.1, loss="ls", max_depth=5, max_features=0.55,
                                  min_samples_leaf=12, min_samples_split=14, n_estimators=100, subsample=0.5)
    )

    exported_pipeline.fit(X_train, y_train)
    print("train:--------------")
    predict = exported_pipeline.predict(X_train)
    error(predict, y_train)

    print("test:---------------")
    predict = exported_pipeline.predict(X_test)
    error(predict, y_test)

    # online
    if online == 1:
        exported_pipeline.fit(train_X, train_Y)
        predict = exported_pipeline.predict(test_X)
        save_to_file(predict, uid, "../result/result_12.11_1_ptot.csv")

def ptot_result(X, Y, Test, uid, online=0):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    test_X = Test.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=RidgeCV()),
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.01, loss="huber", max_depth=6,
                                                              max_features=0.45, min_samples_leaf=12,
                                                              min_samples_split=18, n_estimators=100,
                                                              subsample=0.7500000000000001)),
        RandomForestRegressor(bootstrap=False, max_features=0.05, min_samples_leaf=11, min_samples_split=8,
                              n_estimators=100)
    )

    exported_pipeline.fit(X_train, y_train)
    print("train:--------------")
    predict = exported_pipeline.predict(X_train)
    error(predict, y_train)

    print("test:---------------")
    predict = exported_pipeline.predict(X_test)
    error(predict, y_test)

    # online
    if online == 1:
        exported_pipeline.fit(train_X, train_Y)
        predict = exported_pipeline.predict(test_X)

        save_to_file(predict, uid, "../result/result_12.11_0_ptot.csv")

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


def xgb_train(X, Y, Test, uid, online=0):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    features = [x for x in X.columns]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    model = MyModel()

    # num_round = 150
    bst = xgb.train(model.xgb_r_param, dtrain, model.xgb_r_num_round, evallist)
    # bst = xgb.cv(model.xgb_r_param, dtrain, model.xgb_r_num_round, nfold=5, metrics={'error'}, seed=0, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

    # make prediction
    preds = bst.predict(dtest)
    error(y_test, preds)
    xgb.plot_importance(bst)
    # xgb.plot_tree(bst)
    plt.show()

    if online == 1:
        test_X = Test.as_matrix()

        dtrain = xgb.DMatrix(train_X, label=train_Y)
        dtest = xgb.DMatrix(test_X)
        evallist = [(dtrain, 'train')]
        bst = xgb.train(model.xgb_r_param, dtrain, model.xgb_r_num_round, evallist)
        # make prediction
        predict = bst.predict(dtest)
        save_to_file(predict, uid, "../result/result_12.11_1_gbdt.csv")

    # 加上分类的结果
    # classify = xgb_classify(X, Y)
    # for i in range(len(preds)):
    #     if classify[i] == 0:
    #         preds[i] = 0
    # error(y_test, preds)

    return preds


def offline(X, Y, Test, uid, online=0):
    train_X = X.as_matrix()
    train_Y = Y.as_matrix()

    test_X = Test.as_matrix()

    # train_X = data_scaler(train_X)

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

    clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                    n_estimators=500,
                                    learning_rate=0.05,
                                    max_depth=8,
                                    subsample=0.8,
                                    max_features=0.6,
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

    # online
    if online == 1:
        clf.fit(train_X, train_Y)
        predict = clf.predict(test_X)
        save_to_file(predict, uid, "../result/result_12.11_old_gbdt.csv")


def data_scaler(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data)


def main():

    # X = pd.DataFrame(pd.read_csv("../feature/train_x_offline.csv"))
    # Y = pd.DataFrame(pd.read_csv("../feature/train_y_offline.csv"))
    # Test = pd.DataFrame(pd.read_csv("../feature/test_x_online.csv"))

    X = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_8_end_10.csv"))
    Y = pd.DataFrame(pd.read_csv("../feature/train_y_11_offline.csv"))

    Test = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_8_end_11.csv"))

    # X = X.fillna(0)
    # Test = Test.fillna(0)

    _, uid = delete(X, Test, "uid")
    # delete(X, Test, "average_discount")

    Y.pop("uid")

    # xgb参数
    model = MyModel()

    # classify = xgb_classify(X, Y)

    # offline(X, Y, Test, uid, online = 1)
    # preds = xgb_train(X, Y, Test, uid)
    # xgb_train_online(X, Y, Test, uid)
    # online_GBDT(X, Y, Test, uid)
    # online_LR(X, Y, Test, uid)
    # ptot_result(X, Y, Test, uid, online=1)
    # ptop_2040(X, Y, Test, uid)
    # ptop_1050(X, Y, Test, uid, online=1)
    ptop_2030(X, Y, Test, uid, online=1)


if __name__ == "__main__":
    main()
