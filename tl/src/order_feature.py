from sys import path
path.append('.')
from tl.src.util import split_by_month, count_price_per_order, per_price
import pandas as pd
from math import log


def get_order_feature(MONTH, NUM, order, uid):
    # 提取购物特征
    order["price"] = order["price"].fillna(0)
    # 为消费记录，按照时间分割
    order["month"] = order["buy_time"].apply(split_by_month)
    # 获取用户在每笔费用的实际消费（金钱*数量-折扣）
    order["real_price"] = order.apply(count_price_per_order, axis=1)
    # 用户买的单价
    order["per_price"] = order.apply(per_price, axis=1)

    # 获取每个用户购物平均价格和平均折扣
    average_price = pd.DataFrame(
        {"average_price": order.loc[order["month"] <= MONTH]["real_price"].groupby([order["uid"]]).sum()}).reset_index()
    average_price["average_price"] = average_price["average_price"].apply(lambda x: log(x / NUM + 1, 5))
    average_discount = pd.DataFrame(
        {"average_discount": order.loc[order["month"] <= MONTH]["discount"].groupby(
            [order["uid"]]).sum()}).reset_index()
    average_discount["average_discount"] = average_discount["average_discount"].apply(lambda x: log(x / NUM + 1, 5))
    # 获取每个用户当月的购买物品总价格和
    current_price_sum = pd.DataFrame(
        {"current_price_sum": order.loc[order["month"] == MONTH]["real_price"].groupby(
            [order["uid"]]).sum()}).reset_index()
    current_price_sum["current_price_sum"] = current_price_sum["current_price_sum"].apply(lambda x: log(x + 1, 5))

    # 合并数据集，将User和特征一一对应
    feature = pd.merge(uid, current_price_sum, on=["uid"], how="left")
    feature = pd.merge(feature, average_price, on=["uid"], how="left")
    feature = pd.merge(feature, average_discount, on=["uid"], how="left")

    return feature
