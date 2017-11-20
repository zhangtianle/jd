from dateutil.parser import parse
import numpy as np


def split_by_month(data):
    return parse(data).month


def count_price_per_order(column):
    return column["price"] * column["qty"] - column["discount"]


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

def change_loan(loan):
    return np.round(5**(loan)-1, 2)