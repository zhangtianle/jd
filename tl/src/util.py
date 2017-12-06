from sys import path
path.append('../../')
from sklearn.metrics import mean_squared_error
from dateutil.parser import parse
import numpy as np
import configparser
import pandas as pd


def split_by_month(data):
    return int(data.split('-')[1])
    # return parse(data).month

def split_by_month_further(data, month):
    return parse(data).month > month

def count_price_per_order(column):
    price = column["price"] * column["qty"] - column["discount"]
    if price < 0:
        return 0.0
    return price


def get_pay_per_month(column):
    return column["loan_amount"] / column["plannum"]


def get_remain_loan(column, month):
    tmp = column["loan_amount"] - column["pay_per_month"] * (month - column["month"])
    if tmp >= 0:
        return tmp
    return 0


def get_remain_pay(column, month):
    if month - column["month"] <= column["plannum"] and month - column["month"] > 0:
        return column["pay_per_month"]
    return 0


def per_price(column):
    return (column["price"] + column["discount"]) / column["qty"]


def change_loan(loan):
    return np.round(5 ** loan - 1, 2)


def handle_na(feature):
    return feature.fillna(0.0)


def get_url():
    conf = configparser.ConfigParser()
    conf.read("./jd.conf")

    root_dir = conf.get("local", "root_dir_local")
    train_url = conf.get("local", "train_url")
    feature_url = conf.get("local", "feature_url")
    return root_dir, train_url, feature_url


def read_data():
    root_dir, train_url, feature_url = get_url()
    # read data
    loan = pd.DataFrame(pd.read_csv(root_dir + 't_loan.csv'))
    user = pd.DataFrame(pd.read_csv(root_dir + 't_user.csv'))
    order = pd.DataFrame(pd.read_csv(root_dir + 't_order.csv'))
    click = pd.DataFrame(pd.read_csv(root_dir + 't_click.csv'))

    return loan, user, order, click


def delete(x, test, coloumn):
    return x.pop(coloumn), test.pop(coloumn)


def error(y_train, predict):
    for _ in range(len(predict)):
        if predict[_] < 0:
            predict[_] = 0.0
    print("Mean squared train error: %.6f" % mean_squared_error(y_train, predict) ** 0.5)
