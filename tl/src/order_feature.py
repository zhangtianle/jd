from sys import path

path.append('../../')
from tl.src.util import split_by_month, count_price_per_order, per_price, get_url, read_data, change_loan
import pandas as pd
from math import log
import numpy as np


def get_order_feature(start_month, MONTH, NUM, order, uid):
    # 为消费记录，按照时间分割
    order["month"] = order["buy_time"].apply(split_by_month)
    # 分割
    order = order.loc[(order["month"] >= start_month) & (order["month"] <= MONTH)]
    # 提取购物特征
    order["price"] = order["price"].fillna(0)
    # 获取用户在每笔费用的实际消费（金钱*数量-折扣）price_sum_discount
    order["real_price"] = order.apply(count_price_per_order, axis=1)
    # 用户买的单价
    order["per_price"] = order.apply(per_price, axis=1)
    order["price_sum"] = order.apply(lambda x: x["price"] * x["qty"], axis=1)
    order["free"] = order["price"].apply(lambda x: 1 if x == 0 else 0)
    order["discount_ratio"] = order.apply(lambda x: 1 - x["real_price"] / (x["price_sum"]) if x["price_sum"] != 0 else 0.0, axis=1)

    # 统计所有人平均每次购物的平均价格和平均价格
    static_price_sum_mean = np.array(order["real_price"]).mean()
    static_price_mean = np.array(order["price"]).mean()
    ######
    tmp_price_sum_mean = order.groupby(["uid"]).agg({"real_price": "mean"}).rename(
        columns={"real_price": "price_sum_mean"}).reset_index()

    price_sum_mean = pd.merge(uid, tmp_price_sum_mean, on=["uid"], how="left")
    price_sum_mean = price_sum_mean.fillna(0.0)

    price_sum_mean["price_sum_mean"] = price_sum_mean["price_sum_mean"].apply(lambda x: x - static_price_sum_mean)

    tmp_price_mean = order.groupby(["uid"]).agg({"price": "mean"}).rename(
        columns={"price": "price_mean"}).reset_index()
    price_mean = pd.merge(uid, tmp_price_mean, on=["uid"], how="left")
    price_mean = price_mean.fillna(0.0)
    price_mean["price_mean"] = price_mean["price_mean"].apply(lambda x: x - static_price_mean)

    # 每月最低/最高/平均购买次数
    average_num_order = order.groupby(["uid"]).agg({"uid": "count"}).rename(
        columns={"uid": "average_num_order"}).reset_index()
    average_num_order["average_num_order"] = average_num_order["average_num_order"].apply(lambda x: x / NUM)
    min_max_num_order = pd.DataFrame(
        order["real_price"].groupby([order["uid"], order["month"]]).count()).reset_index()
    min_num_order = pd.DataFrame(
        min_max_num_order["real_price"].groupby([min_max_num_order["uid"]]).min()).rename(
        columns={"real_price": "min_num_order"}).reset_index()
    max_num_order = pd.DataFrame(
        min_max_num_order["real_price"].groupby([min_max_num_order["uid"]]).max()).rename(
        columns={"real_price": "max_num_order"}).reset_index()
    std_num_order = order.groupby(["uid", "month"]).agg({"uid": "count"}).rename(
        columns={"uid": "std_num_order"}).reset_index()
    std_num_order = std_num_order.groupby(["uid"]).agg({"std_num_order": "std"}).reset_index()

    # 获取每个用户每次消费的平均/最低/最高的价格
    average_price_each = order.groupby(["uid"]).agg({"real_price": "mean"}).rename(
        columns={"real_price": "average_price_each"}).reset_index()
    average_price_each["average_price_each"] = average_price_each["average_price_each"].apply(lambda x: log(x + 1, 5))
    min_price_each = pd.DataFrame({"min_price_each": order["real_price"].groupby([order["uid"]]).min()}).reset_index()
    min_price_each["min_price_each"] = min_price_each["min_price_each"].apply(lambda x: log(x + 1, 5))
    max_price_each = pd.DataFrame({"max_price_each": order["real_price"].groupby([order["uid"]]).max()}).reset_index()
    max_price_each["max_price_each"] = max_price_each["max_price_each"].apply(lambda x: log(x + 1, 5))
    sum_price_each = pd.DataFrame({"sum_price_each": order["real_price"].groupby([order["uid"]]).sum()}).reset_index()
    sum_price_each["sum_price_each"] = sum_price_each["sum_price_each"].apply(lambda x: log(x + 1, 5))
    std_price_each = order.groupby(["uid"]).agg({"real_price": "std"}).rename(
        columns={"real_price": "std_price_each"}).reset_index()

    # 获取每个用户购物平均价格
    average_price = pd.DataFrame({"average_price": order["real_price"].groupby([order["uid"]]).sum()}).reset_index()
    average_price["average_price"] = average_price["average_price"].apply(lambda x: log(x / NUM + 1, 5))
    min_max_price = pd.DataFrame(order["price_sum"].groupby([order["uid"], order["month"]]).sum()).reset_index()

    min_price_month = pd.DataFrame(min_max_price["price_sum"].groupby([min_max_price["uid"]]).min()).reset_index()
    min_price_month = min_price_month.rename(columns={"price_sum": "min_price_month"})
    min_price_month["min_price_month"] = min_price_month["min_price_month"].apply(lambda x: log(x + 1, 5))

    max_price_month = pd.DataFrame(min_max_price["price_sum"].groupby([min_max_price["uid"]]).max()).reset_index()
    max_price_month = max_price_month.rename(columns={"price_sum": "max_price_month"})
    max_price_month["max_price_month"] = max_price_month["max_price_month"].apply(lambda x: log(x + 1, 5))

    std_price_sum = order["price_sum"].groupby([order["uid"], order["month"]]).sum().reset_index()
    std_price_sum = std_price_sum.groupby(["uid"]).agg({"price_sum": "std"}).rename(
        columns={"price_sum": "std_price_sum"}).reset_index()

    # 获取每个用户购物折扣后总价：平均/最低/最高
    average_real_price = pd.DataFrame(
        {"average_real_price": order["real_price"].groupby([order["uid"]]).sum()}).reset_index()
    average_real_price["average_real_price"] = average_real_price[
        "average_real_price"].apply(lambda x: log(x / NUM + 1, 5))
    min_max_real_price = pd.DataFrame(
        order["real_price"].groupby([order["uid"], order["month"]]).sum()).reset_index()
    min_real_price_month = pd.DataFrame(min_max_real_price["real_price"].groupby(
        [min_max_real_price["uid"]]).min()).reset_index()
    min_real_price_month = min_real_price_month.rename(
        columns={"real_price": "min_real_price_month"})
    min_real_price_month["min_real_price_month"] = min_real_price_month[
        "min_real_price_month"].apply(lambda x: log(x + 1, 5))
    max_real_price_month = pd.DataFrame(min_max_real_price["real_price"].groupby(
        [min_max_real_price["uid"]]).max()).reset_index()
    max_real_price_month = max_real_price_month.rename(
        columns={"real_price": "max_real_price_month"})
    max_real_price_month["max_real_price_month"] = max_real_price_month[
        "max_real_price_month"].apply(lambda x: log(x + 1, 5))

    std_real_price = order["real_price"].groupby(
        [order["uid"], order["month"]]).sum().reset_index()
    std_real_price = std_real_price.groupby(["uid"]).agg({"real_price": "std"}).rename(
        columns={"price_sum": "std_real_price"}).reset_index()

    # 平均折扣
    average_discount = pd.DataFrame({"average_discount": order["discount"].groupby([order["uid"]]).sum()}).reset_index()
    average_discount["average_discount"] = average_discount["average_discount"].apply(lambda x: log(x / NUM + 1, 5))

    # 获取每个用户每次购买平均/最大/最小购买量
    average_qty_each = order.groupby(["uid"]).agg({"qty": "mean"}).rename(
        columns={"qty": "average_qty_each"}).reset_index()
    min_qty = average_qty_each["average_qty_each"].min()
    max_qty = average_qty_each["average_qty_each"].max()
    average_qty_each["average_qty_each"] = average_qty_each["average_qty_each"].apply(
        lambda x: x - min_qty / (max_qty - min_qty))
    min_qty_each = pd.DataFrame({"min_qty_each": order["qty"].groupby([order["uid"]]).min()}).reset_index()
    min_qty = min_qty_each["min_qty_each"].min()
    max_qty = min_qty_each["min_qty_each"].max()
    min_qty_each["min_qty_each"] = min_qty_each["min_qty_each"].apply(lambda x: x - min_qty / (max_qty - min_qty))
    max_qty_each = pd.DataFrame({"max_qty_each": order["qty"].groupby([order["uid"]]).max()}).reset_index()
    min_qty = max_qty_each["max_qty_each"].min()
    max_qty = max_qty_each["max_qty_each"].max()
    max_qty_each["max_qty_each"] = max_qty_each["max_qty_each"].apply(lambda x: x - min_qty / (max_qty - min_qty))

    std_qty_each = order.groupby(["uid"]).agg({"qty": "std"}).rename(
        columns={"qty": "std_qty_each"}).reset_index()

    # 历史平均折扣力度(discount/总价格)
    average_discount_price_ration = pd.DataFrame(
        {"average_discount_price_ration": order["discount_ratio"].groupby([order["uid"]]).mean()}).reset_index()
    std_discount_price_ration = pd.DataFrame(
        {"std_discount_price_ration": order["discount_ratio"].groupby([order["uid"]]).std()}).reset_index()

    # 平均每月免费与非免费的占比
    # average_free_no_free_percentage = order.groupby(["uid"]).agg({"free": "sum",
    #                                                               "uid": "count"}).rename(
    #     columns={"free": "average_free_percentage", "uid": "uid_count"}).reset_index()
    # average_free_no_free_percentage["average_free_percentage"] = average_free_no_free_percentage.apply(
    #     lambda x: x["average_free_percentage"] / x["uid_count"] / NUM, axis=1)
    # average_free_no_free_percentage["avaerage_no_free_percentage"] = average_free_no_free_percentage.apply(
    #     lambda x: 1.0 - x["average_free_percentage"] / NUM, axis=1)
    # average_free_no_free_percentage.pop("uid_count")
    # 平均/最低/最高单价
    # average_single_price = order.groupby(["uid"]).agg({"price": "mean"}).rename(
    #     columns={"price": "average_single_price"}).reset_index()
    # average_single_price["average_single_price"] = average_single_price[
    #     "average_single_price"].apply(lambda x: log(x + 1, 5))
    # max_single_price = order.groupby(["uid"]).agg({"price": "max"}).rename(
    #     columns={"price": "max_single_price"}).reset_index()
    # max_single_price["max_single_price"] = max_single_price[
    #     "max_single_price"].apply(lambda x: log(x + 1, 5))
    # min_single_price = order.groupby(["uid"]).agg({"price": "min"}).rename(
    #     columns={"price": "min_single_price"}).reset_index()
    # min_single_price["min_single_price"] = min_single_price[
    #     "min_single_price"].apply(lambda x: log(x + 1, 5))
    #
    # std_single_price = order.groupby(["uid"]).agg({"price": "std"}).rename(
    #     columns={"price": "std_single_price"}).reset_index()

    # 每月非免费平均总价格/每月非免费平均折扣后的总价/每月非免费平均折扣力度/每月非免费平均折扣
    average_no_free_discount_ratio = pd.DataFrame(
        {"average_no_free_discount_ratio": order.loc[order["free"] != 1]["discount_ratio"].groupby(
            [order["uid"]]).mean()}).reset_index()
    std_no_free_discount_ratio = pd.DataFrame(
        {"std_no_free_discount_ratio": order.loc[order["free"] != 1]["discount_ratio"].groupby(
            [order["uid"]]).std()}).reset_index()

    average_no_free_price_discount_sum_mean = pd.DataFrame(
        {"average_no_free_price_discount_sum_mean": order.loc[order["free"] != 1][
            "real_price"].groupby([order["uid"]]).mean()}).reset_index()
    average_no_free_price_discount_sum_mean["average_no_free_price_discount_sum_mean"] = average_no_free_price_discount_sum_mean["average_no_free_price_discount_sum_mean"].apply(lambda x: log(x + 1, 5))

    std_no_free_price_discount_sum_mean = pd.DataFrame(
        {"std_no_free_price_discount_sum_mean": order.loc[order["free"] != 1][
            "real_price"].groupby([order["uid"]]).std()}).reset_index()

    average_no_free_discount_mean = pd.DataFrame({"average_no_free_discount_mean": order.loc[order["free"] != 1][
        "discount"].groupby([order["uid"]]).mean()}).reset_index()
    average_no_free_discount_mean["average_no_free_discount_mean"] = average_no_free_discount_mean[
        "average_no_free_discount_mean"].apply(lambda x: log(x + 1, 5))
    std_no_free_discount_mean = pd.DataFrame({"std_no_free_discount_mean": order.loc[order["free"] != 1][
        "discount"].groupby([order["uid"]]).std()}).reset_index()

    average_no_free_price_sum_mean = pd.DataFrame({"average_no_free_price_sum_mean": order.loc[order["free"] != 1][
        "price_sum"].groupby([order["uid"]]).mean()}).reset_index()
    average_no_free_price_sum_mean["average_no_free_price_sum_mean"] = average_no_free_price_sum_mean[
        "average_no_free_price_sum_mean"].apply(lambda x: log(x + 1, 5))

    std_no_free_price_sum_mean = pd.DataFrame({"std_no_free_price_sum_mean": order.loc[order["free"] != 1][
        "price_sum"].groupby([order["uid"]]).std()}).reset_index()

    average_no_free_single_price_mean = pd.DataFrame({"average_no_free_single_price_mean":
                                                          order.loc[order["free"] != 1]["price"].groupby(
                                                              [order["uid"]]).mean()}).reset_index()
    average_no_free_single_price_mean["average_no_free_single_price_mean"] = average_no_free_single_price_mean[
        "average_no_free_single_price_mean"].apply(lambda x: log(x + 1, 5))

    std_no_free_single_price_mean = pd.DataFrame({"std_no_free_single_price_mean": order.loc[order["free"] != 1][
        "price"].groupby([order["uid"]]).std()}).reset_index()

    # 获取每个用户当月的购买物品总价格和
    current = order.loc[order["month"] == MONTH]
    current_price_sum = pd.DataFrame(
        {"current_price_sum": current["real_price"].groupby([current["uid"]]).sum()}).reset_index()
    current_price_sum["current_price_sum"] = current_price_sum["current_price_sum"].apply(lambda x: log(x + 1, 5))
    current_real_price_average = pd.DataFrame({"current_real_price_average": current[
        "real_price"].groupby([current["uid"]]).mean()}).reset_index()
    current_real_price_average["current_real_price_average"] = current_real_price_average[
        "current_real_price_average"].apply(lambda x: log(x + 1, 5))
    # 获取每个用户当月的总消费次数、免费与非免费的比例
    # current_num_order = current.groupby(["uid"]).agg({"free": "sum",
    #                                                   "uid": "count"}).rename(
    #     columns={"free": "current_free_percentage", "uid": "current_order_num"}).reset_index()
    # current_num_order["current_free_percentage"] = current_num_order.apply(
    #     lambda x: x["current_free_percentage"] / x["current_order_num"], axis=1)
    # current_num_order["current_no_free_percentage"] = current_num_order.apply(
    #     lambda x: 1.0 - x["current_free_percentage"], axis=1)

    current_no_free_price_sum_mean = pd.DataFrame({"current_no_free_price_sum_mean": current.loc[current["free"] != 1][
        "price_sum"].groupby([current["uid"]]).mean()}).reset_index()
    current_no_free_price_sum_mean["current_no_free_price_sum_mean"] = current_no_free_price_sum_mean[
        "current_no_free_price_sum_mean"].apply(lambda x: log(x + 1, 5))

    current_no_free_price_discount_sum_mean = pd.DataFrame(
        {"current_no_free_price_discount_sum_mean": current.loc[current["free"] != 1][
            "real_price"].groupby([current["uid"]]).mean()}).reset_index()
    current_no_free_price_discount_sum_mean["current_no_free_price_discount_sum_mean"] = \
        current_no_free_price_discount_sum_mean[
            "current_no_free_price_discount_sum_mean"].apply(lambda x: log(x + 1, 5))

    current_no_free_discount_mean = pd.DataFrame({"current_no_free_discount_mean": current.loc[current["free"] != 1][
        "discount"].groupby([current["uid"]]).mean()}).reset_index()
    current_no_free_discount_mean["current_no_free_discount_mean"] = current_no_free_discount_mean[
        "current_no_free_discount_mean"].apply(lambda x: log(x + 1, 5))

    current_no_free_discount_ratio_mean = pd.DataFrame(
        {"current_no_free_discount_ratio_mean": current.loc[current["free"] != 1][
            "discount_ratio"].groupby([current["uid"]]).mean()}).reset_index()
    # 当月平均单价
    current_price_mean = current.groupby(["uid"]).agg({"price": "mean"}).rename(
        columns={"price": "current_price_mean"}).reset_index()
    current_price_mean["current_price_mean"] = current_price_mean[
        "current_price_mean"].apply(lambda x: log(x + 1, 5))

    std_current_price_mean = current.groupby(["uid"]).agg({"price": "std"}).rename(
        columns={"price": "std_current_price_mean"}).reset_index()

    # 当月非免费平均单价
    current_no_freeprice_mean = current.loc[current["free"] != 1].groupby(["uid"]).agg({"price": "mean"}).rename(
        columns={"price": "current_no_freeprice_mean"}).reset_index()
    current_no_freeprice_mean["current_no_freeprice_mean"] = current_no_freeprice_mean[
        "current_no_freeprice_mean"].apply(lambda x: log(x + 1, 5))
    std_current_no_freeprice_mean = current.loc[current["free"] != 1].groupby(["uid"]).agg({"price": "std"}).rename(
        columns={"price": "std_current_no_freeprice_mean"}).reset_index()
    # 当月每次平均折扣/每次折扣力度
    current_discount_mean = pd.DataFrame({"current_discount_mean": current[
        "discount"].groupby([current["uid"]]).mean()}).reset_index()
    current_discount_mean["current_discount_mean"] = current_discount_mean[
        "current_discount_mean"].apply(lambda x: log(x + 1, 5))
    std_current_discount_mean = pd.DataFrame({"std_current_discount_mean": current[
        "discount"].groupby([current["uid"]]).std()}).reset_index()

    current_discount_ratio_mean = pd.DataFrame({"current_discount_ratio_mean": current[
        "discount_ratio"].groupby([current["uid"]]).mean()}).reset_index()
    std_current_discount_ratio_mean = pd.DataFrame({"std_current_discount_ratio_mean": current[
        "discount_ratio"].groupby([current["uid"]]).std()}).reset_index()

    features = [average_price_each, min_price_each, max_price_each, sum_price_each, average_discount,
                average_discount_price_ration,
                average_price,
                max_price_month,
                min_price_month,
                average_num_order,
                min_num_order,
                max_num_order,
                average_qty_each,
                min_qty_each,
                max_qty_each,
                # average_free_no_free_percentage,
                current_price_sum,
                current_real_price_average,
                std_current_price_mean,
                # current_num_order,
                current_no_free_price_sum_mean,
                current_no_free_price_discount_sum_mean,
                current_no_free_discount_mean,
                current_no_free_discount_ratio_mean,
                current_price_mean,
                # average_single_price,
                average_no_free_discount_ratio,
                average_no_free_price_discount_sum_mean,
                average_no_free_discount_mean,
                average_no_free_price_sum_mean,
                # max_single_price,
                # min_single_price,
                average_real_price,
                min_real_price_month,
                max_real_price_month,
                average_no_free_single_price_mean,
                current_discount_mean,
                current_discount_ratio_mean,
                current_no_freeprice_mean,
                std_current_discount_ratio_mean,
                std_current_discount_mean,
                std_current_price_mean,
                std_current_no_freeprice_mean,
                std_no_free_single_price_mean,
                std_no_free_price_sum_mean,
                std_no_free_discount_mean,
                std_no_free_price_discount_sum_mean,
                std_no_free_discount_ratio,
                # std_single_price,
                std_discount_price_ration,
                std_real_price,
                std_qty_each,
                std_price_sum,
                std_price_each,
                std_num_order]
                # price_mean]
                # price_sum_mean]

    # 合并数据集，将User和特征一一对应
    result_feature = pd.DataFrame({"uid": uid['uid']})

    for feature in features:
        result_feature = pd.merge(result_feature, feature, on=["uid"], how="left")

    return result_feature

if __name__ == '__main__':
    root_dir, train_url, feature_url = get_url()
    loan, user, order, click = read_data()

    # 转换金额
    loan['loan_amount'] = change_loan(loan['loan_amount'])
    order['price'] = change_loan(order['price'])
    order['discount'] = change_loan(order['discount'])

    uid = pd.DataFrame(user["uid"])

    for start_month in [8]:
        MONTH = start_month + 2
        NUM = 3
        feature = get_order_feature(start_month, MONTH, NUM, order, uid)

        feature.to_csv(feature_url + 'order_feature_start_{0}_end_{1}.csv'.format(start_month, MONTH), index=False)