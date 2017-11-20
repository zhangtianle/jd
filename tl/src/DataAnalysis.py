import pandas as pd
import configparser
from tl.src.util import *
from math import log

conf = configparser.ConfigParser()
conf.read("./jd.conf")

root_dir = conf.get("local", "root_dir_local")
train_url = conf.get("local", "train_url")
feature_url = conf.get("local", "feature_url")

# read data
loan = pd.DataFrame(pd.read_csv(root_dir + 't_loan.csv'))
user = pd.DataFrame(pd.read_csv(root_dir + 't_user.csv'))
order = pd.DataFrame(pd.read_csv(root_dir + 't_order.csv'))

loan['loan_amount'] = change_loan(loan['loan_amount'])
order['price'] = change_loan(order['price'])
order['discount'] = change_loan(order['discount'])
order.to_csv(feature_url + 'order.csv')

uid = pd.DataFrame(user["uid"])

MONTH = 10
NUM = 3.0
# 提取历史贷款信息
loan["month"] = loan["loan_time"].apply(split_by_month)
# 特征（逐个计算）
loan["pay_per_month"] = loan.apply(get_pay_per_month, axis=1)
loan["remain_loan"] = loan.loc[loan["month"] <= MONTH].apply(get_remain_loan, axis=1, args=(MONTH,))
loan["remain_loan"] = loan["remain_loan"].fillna(0)
loan["remain_pay"] = loan.loc[loan["month"] <= MONTH].apply(get_remain_pay, axis=1, args=(MONTH,))
loan["remain_pay"] = loan["remain_pay"].fillna(0)
# 汇总
# 平均每月贷款
average_loan = pd.DataFrame(
    {"average_loan": loan.loc[loan["month"] <= MONTH]["loan_amount"].groupby([loan["uid"]]).sum()}).reset_index()
average_loan["average_loan"] = average_loan["average_loan"].apply(lambda x: log(x / NUM + 1, 5))
# 平均每月月供
average_pay = pd.DataFrame(
    {"average_pay": loan.loc[loan["month"] <= MONTH]["pay_per_month"].groupby([loan["uid"]]).sum()}).reset_index()
average_pay["average_pay"] = average_pay["average_pay"].apply(lambda x: log(x / NUM + 1, 5))
# 历史贷款总额
remain_loan = pd.DataFrame(loan.loc[loan["month"] <= MONTH]["remain_loan"].groupby([loan["uid"]]).sum()).reset_index()
remain_loan["remain_loan"] = remain_loan["remain_loan"].apply(lambda x: log(x + 1, 5))
# 累计月供
remain_pay = pd.DataFrame(loan.loc[loan["month"] <= MONTH]["remain_pay"].groupby([loan["uid"]]).sum()).reset_index()
remain_pay["remain_pay"] = remain_pay["remain_pay"].apply(lambda x: log(x + 1, 5))
# 当月月供和当月贷款总额
current_loan_sum = pd.DataFrame(
    {"current_loan_sum": loan.loc[loan["month"] == MONTH]["loan_amount"].groupby([loan["uid"]]).sum()}).reset_index()
current_loan_sum["current_loan_sum"] = current_loan_sum["current_loan_sum"].apply(lambda x: log(x + 1, 5))
current_pay_sum = pd.DataFrame(
    {"current_pay_sum": loan.loc[loan["month"] == MONTH]["pay_per_month"].groupby([loan["uid"]]).sum()}).reset_index()
current_pay_sum["current_pay_sum"] = current_pay_sum["current_pay_sum"].apply(lambda x: log(x + 1, 5))

feature_loan = pd.merge(uid, average_loan, on=["uid"], how="left")
feature_loan = pd.merge(feature_loan, average_pay, on=["uid"], how="left")
feature_loan = pd.merge(feature_loan, remain_loan, on=["uid"], how="left")
feature_loan = pd.merge(feature_loan, remain_pay, on=["uid"], how="left")
feature_loan = pd.merge(feature_loan, current_pay_sum, on=["uid"], how="left")
feature_loan = pd.merge(feature_loan, current_loan_sum, on=["uid"], how="left")

# 提取购物特征
order["price"] = order["price"].fillna(0)
# 为消费记录，按照时间分割
order["month"] = order["buy_time"].apply(split_by_month)
# 获取用户在每笔费用的实际消费（金钱*数量-折扣）
order["real_price"] = order.apply(count_price_per_order, axis=1)

# 获取每个用户购物平均价格和平均折扣
average_price = pd.DataFrame(
    {"average_price": order.loc[order["month"] <= MONTH]["real_price"].groupby([order["uid"]]).sum()}).reset_index()
average_price["average_price"] = average_price["average_price"].apply(lambda x: log(x / NUM + 1, 5))
average_discount = pd.DataFrame(
    {"average_discount": order.loc[order["month"] <= MONTH]["discount"].groupby([order["uid"]]).sum()}).reset_index()
average_discount["average_discount"] = average_discount["average_discount"].apply(lambda x: log(x / NUM + 1, 5))
# 获取每个用户当月的购买物品总价格和
current_price_sum = pd.DataFrame(
    {"current_price_sum": order.loc[order["month"] == MONTH]["real_price"].groupby([order["uid"]]).sum()}).reset_index()
current_price_sum["current_price_sum"] = current_price_sum["current_price_sum"].apply(lambda x: log(x + 1, 5))

# 合并数据集，将User和特征一一对应
feature = pd.merge(uid, user, on=["uid"], how="left")
feature = pd.merge(feature, current_price_sum, on=["uid"], how="left")
feature = pd.merge(feature, average_price, on=["uid"], how="left")
feature = pd.merge(feature, average_discount, on=["uid"], how="left")
feature = pd.merge(feature, feature_loan, on=["uid"], how="left")

# 处理异常值
feature["current_pay_sum"] = feature["current_pay_sum"].fillna(0.0)
feature["current_loan_sum"] = feature["current_loan_sum"].fillna(0.0)
feature["current_price_sum"] = feature["current_price_sum"].fillna(0.0)
feature["average_price"] = feature["average_price"].fillna(0.0)
feature["average_loan"] = feature["average_loan"].fillna(0.0)
feature["average_pay"] = feature["average_pay"].fillna(0.0)
feature["remain_loan"] = feature["remain_loan"].fillna(0.0)
feature["remain_pay"] = feature["remain_pay"].fillna(0.0)
feature["average_discount"] = feature["average_discount"].fillna(0.0)
# 保存特征数据
feature.to_csv("../train/train_x_offline.csv", index=False)

# 获得预测值
loan_next_month = pd.DataFrame(pd.read_csv(root_dir + "t_loan_sum.csv"))
loan_next_month.pop("month")

loan_next_month = pd.merge(uid, loan_next_month, on=["uid"], how="left")
loan_next_month["loan_sum"] = loan_next_month["loan_sum"].fillna(0.0)
# 保存预测数据
loan_next_month.to_csv(feature_url + "train_y_offline.csv", index=False)
