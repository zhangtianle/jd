from sys import path
path.append('../../')
from tl.src.util import count_price_per_order, get_url, read_data, change_loan, split_by_month
import pandas as pd
from math import log


def get_order_loan(order, loan, uid, start_month, MONTH, NUM):
    order["month"] = order["buy_time"].apply(split_by_month)
    loan["month"] = loan["loan_time"].apply(split_by_month)
    order = order.loc[(order["month"] >= start_month) & (order["month"] <= MONTH)]
    loan = loan.loc[(loan["month"] >= start_month) & (loan["month"] <= MONTH)]

    order["real_price"] = order.apply(count_price_per_order, axis=1)
    current_price_sum = pd.DataFrame({"current_price_sum": order.loc[order["month"] == MONTH]["real_price"].groupby(
        [order["uid"]]).sum()}).reset_index()
    current_price_sum["current_price_sum"] = current_price_sum["current_price_sum"].apply(lambda x: log(x + 1, 5))

    current_loan_sum = pd.DataFrame({"current_loan_sum": loan.loc[loan["month"] == MONTH]["loan_amount"].groupby(
        [loan["uid"]]).sum()}).reset_index()
    current_loan_sum["current_loan_sum"] = current_loan_sum["current_loan_sum"].apply(lambda x: log(x + 1, 5))

    loan_order = pd.merge(current_price_sum, current_loan_sum, on="uid", how="left")
    loan_order = loan_order.fillna(0)
    loan_order["loan_order_ratio"] = loan_order.apply(lambda x: x["current_loan_sum"] / (x["current_price_sum"] + 1), axis=1)

    loan_order = pd.merge(uid, loan_order, on="uid", how="left")

    # 获取每个用户当月的购买物品总价格和
    price_sum = pd.DataFrame({"price_sum": order["real_price"].groupby([order["uid"]]).sum()}).reset_index()

    # 期间内贷款总额
    loan_sum = pd.DataFrame({"loan_sum": loan["loan_amount"].groupby([loan["uid"]]).sum()}).reset_index()

    # 期间贷款总额与购买差值
    order_loan = pd.merge(left=loan_sum, right=price_sum, on="uid", how="left")
    order_loan['diff_order_loan'] = order_loan.apply(lambda x: x["price_sum"] - x["loan_sum"], axis=1)
    min_diff = order_loan['diff_order_loan'].min()
    order_loan['diff_order_loan'] = order_loan['diff_order_loan'].apply(lambda x: log(x - min_diff + 1, 5))

    loan_order = pd.merge(left=loan_order, right=order_loan, on="uid", how="left")

    loan_order = pd.DataFrame(loan_order[["uid", "loan_order_ratio", "diff_order_loan"]])
    return loan_order

if __name__ == '__main__':
    root_dir, train_url, feature_url = get_url()
    loan, user, order, click = read_data()

    # 转换金额
    loan['loan_amount'] = change_loan(loan['loan_amount'])
    order['price'] = change_loan(order['price'])
    order['discount'] = change_loan(order['discount'])

    uid = pd.DataFrame(user["uid"])

    for start_month in [8]:
        MONTH = start_month + 3
        NUM = 4
        order_loan = get_order_loan(order, loan, uid, start_month, MONTH, NUM)

        order_loan.to_csv(feature_url + 'order_loan_feature_start_{0}_end_{1}.csv'.format(start_month, MONTH), index=False)