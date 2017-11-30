from sys import path

path.append('.')
from tl.src.util import split_by_month, count_price_per_order, per_price
import pandas as pd
from math import log


def get_order_feature(start_month, MONTH, NUM, order, uid):
    # 为消费记录，按照时间分割
    order["month"] = order["buy_time"].apply(split_by_month)
    # 分割
    order = order[(order["month"] >= start_month) & (order["month"] <= MONTH)]
    # 提取购物特征
    order["price"] = order["price"].fillna(0)
    # 获取用户在每笔费用的实际消费（金钱*数量-折扣）
    order["real_price"] = order.apply(count_price_per_order, axis=1)
    # 用户买的单价
    order["per_price"] = order.apply(per_price, axis=1)

    # 获取每个用户每次消费的平均/最低/最高的价格
    average_price_each = order.groupby(["uid"]).agg({"real_price": "mean"}).rename(
        columns={"real_price": "average_price_each"}).reset_index()
    average_price_each["average_price_each"] = average_price_each["average_price_each"].apply(lambda x: log(x + 1, 5))
    min_price_each = pd.DataFrame(
        {"min_price_each": order["real_price"].groupby([order["uid"]]).min()}).reset_index()
    min_price_each["min_price_each"] = min_price_each["min_price_each"].apply(lambda x: log(x + 1, 5))
    max_price_each = pd.DataFrame(
        {"max_price_each": order["real_price"].groupby([order["uid"]]).max()}).reset_index()
    max_price_each["max_price_each"] = max_price_each["max_price_each"].apply(lambda x: log(x + 1, 5))

    # 获取每个用户购物平均价格和平均折扣
    average_price = pd.DataFrame(
        {"average_price": order["real_price"].groupby([order["uid"]]).sum()}).reset_index()
    average_price["average_price"] = average_price["average_price"].apply(lambda x: log(x / NUM + 1, 5))
    average_discount = pd.DataFrame(
        {"average_discount": order["discount"].groupby(
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
    feature = pd.merge(feature, average_price_each, on=["uid"], how="left")
    feature = pd.merge(feature, min_price_each, on=["uid"], how="left")
    feature = pd.merge(feature, max_price_each, on=["uid"], how="left")

    return feature
