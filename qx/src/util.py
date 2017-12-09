import Load
from dateutil.parser import parse
import pandas as pd
import numpy as np
from math import log
import datetime as dt
import datetime
import copy as cp

#获取历史统计数据
def split_by_month(data):
    return parse(data).month

def count_price_per_order(column):
    price = column["price"]*column["qty"] - column["discount"]
    if price < 0:
        return 0.0
    return price

def get_pay_per_month(column):
    return column["loan_amount"] / column["plannum"]

def get_remain_loan(column, month):
    tmp = column["loan_amount"] - column["pay_per_month"] * (month - column["month"])
    if tmp >=0 :
        return tmp
    return 0
def get_remain_pay(column, month):
    if month - column["month"] <= column["plannum"] and  month - column["month"] > 0 :
        return column["pay_per_month"]
    return 0

def count_continuous_loan(column, month):
    count = -1
    max_count = 0
    for i in range(8,month+1):
        if column[str(i)] > 0:
            count +=1
        else:
            if max_count < count:
                max_count = count
            count = -1
    return max_count

def get_over(x, array):
    count = 0.0
    for i in range(len(array)):
        if x > array[i]:
            count +=1.0
        else:
            return count / len(array)
    return count / len(array)

def get_order_feature(order, MONTH, NUM, uid):
    history = order.loc[order["month"] <= MONTH]
    
	
	
	#统计所有人平均每次购物的平均价格和平均价格
    static_price_sum_mean = np.array(history["price_sum_discount"]).mean()
    static_price_mean= np.array(history["price"]).mean()
    #######
    tmp_price_sum_mean = history.groupby(["uid"]).agg({"price_sum_discount": "mean"}).rename(
        columns={"price_sum_discount": "price_sum_mean"}).reset_index()

    price_sum_mean = pd.DataFrame({"uid":uid})
    price_sum_mean = pd.merge(price_sum_mean, tmp_price_sum_mean, on=["uid"], how="left")
    price_sum_mean = price_sum_mean.fillna(0.0)
    #tmp_price_sum_mean_array = np.array(price_sum_mean["price_sum_mean"])
    #tmp_price_sum_mean_array.sort()
    #price_sum_mean["price_sum_mean_over"] = price_sum_mean["price_sum_mean"].apply(lambda x : get_over(x, tmp_price_sum_mean_array))
    price_sum_mean["price_sum_mean"] = price_sum_mean["price_sum_mean"].apply(lambda x: x - static_price_sum_mean)
    #####
    tmp_price_mean = history.groupby(["uid"]).agg({"price": "mean"}).rename(
        columns={"price": "price_mean"}).reset_index()
    price_mean = pd.DataFrame({"uid":uid})
    price_mean = pd.merge(price_mean, tmp_price_mean, on=["uid"], how="left")
    price_mean = price_mean.fillna(0.0)
    #tmp_price_mean_array = np.array(price_mean["price_mean"]).sort()
    #price_mean["price_mean_over"] = price_mean["price_mean"].apply(lambda x : get_over(x, tmp_price_mean_array))
    price_mean["price_mean"] = price_mean["price_mean"].apply(lambda x : x - static_price_mean)

    
	
	
	
	#每月最低/最高/平均购买次数
    average_num_order = history.groupby(["uid"]).agg({"uid": "count"}).rename(
        columns={"uid": "average_num_order"}).reset_index()
    average_num_order["average_num_order"] = average_num_order["average_num_order"].apply(lambda x: x / NUM)
    min_max_num_order = pd.DataFrame(
        history["price_sum_discount"].groupby([history["uid"], history["month"]]).count()).reset_index()
    min_num_order = pd.DataFrame(min_max_num_order["price_sum_discount"].groupby([min_max_num_order["uid"]]).min()).rename(
        columns={"price_sum_discount": "min_num_order"}).reset_index()
    max_num_order = pd.DataFrame(min_max_num_order["price_sum_discount"].groupby([min_max_num_order["uid"]]).max()).rename(
        columns={"price_sum_discount": "max_num_order"}).reset_index()


    std_num_order = history.groupby(["uid","month"]).agg({"uid": "count"}).rename(
        columns={"uid": "std_num_order"}).reset_index()
    std_num_order = std_num_order.groupby(["uid"]).agg({"std_num_order":"std"}).reset_index()


    #获取每个用户每次消费的平均/最低/最高的
    average_price_each = history.groupby(["uid"]).agg({"price_sum_discount":"mean"}).rename(columns={"price_sum_discount":"average_price_each"}).reset_index()
    average_price_each["average_price_each"] = average_price_each["average_price_each"].apply(lambda x: log(x + 1, 5))
    min_price_each = pd.DataFrame({"min_price_each" : history["price"].groupby([history["uid"]]).min()}).reset_index()
    min_price_each["min_price_each"] = min_price_each["min_price_each"].apply(lambda x: log(x + 1, 5))
    max_price_each = pd.DataFrame({"max_price_each": history["price"].groupby([history["uid"]]).max()}).reset_index()
    max_price_each["max_price_each"] = max_price_each["max_price_each"].apply(lambda x: log(x + 1, 5))

    std_price_each = history.groupby(["uid"]).agg({"price_sum_discount": "std"}).rename(
        columns={"price_sum_discount": "std_price_each"}).reset_index()

    #获取每个用户购物总价：平均/最低/最高
    average_price = pd.DataFrame({"average_price":history["price_sum"].groupby([history["uid"]]).sum()}).reset_index()
    average_price["average_price"] = average_price["average_price"].apply(lambda x: log(x/NUM + 1, 5))
    min_max_price = pd.DataFrame(history["price_sum"].groupby([history["uid"],history["month"]]).sum()).reset_index()

    min_price_month = pd.DataFrame(min_max_price["price_sum"].groupby([min_max_price["uid"]]).min()).reset_index()
    min_price_month = min_price_month.rename(columns={"price_sum":"min_price_month"})
    min_price_month["min_price_month"] = min_price_month["min_price_month"].apply(lambda x: log(x + 1, 5))

    max_price_month = pd.DataFrame(min_max_price["price_sum"].groupby([min_max_price["uid"]]).max()).reset_index()
    max_price_month = max_price_month.rename(columns={"price_sum":"max_price_month"})
    max_price_month["max_price_month"] = max_price_month["max_price_month"].apply(lambda x: log(x + 1, 5))

    std_price_sum = history["price_sum"].groupby([history["uid"],history["month"]]).sum().reset_index()
    std_price_sum = std_price_sum.groupby(["uid"]).agg({"price_sum":"std"}).rename(columns={"price_sum":"std_price_sum"}).reset_index()

    #获取每个用户购物折扣后总价：平均/最低/最高
    average_price_sum_discount = pd.DataFrame({"average_price_sum_discount":history["price_sum_discount"].groupby([history["uid"]]).sum()}).reset_index()
    average_price_sum_discount["average_price_sum_discount"] = average_price_sum_discount["average_price_sum_discount"].apply(lambda x: log(x/NUM + 1, 5))
    min_max_price_sum_discount = pd.DataFrame(history["price_sum_discount"].groupby([history["uid"],history["month"]]).sum()).reset_index()
    min_price_sum_discount_month = pd.DataFrame(min_max_price_sum_discount["price_sum_discount"].groupby([min_max_price_sum_discount["uid"]]).min()).reset_index()
    min_price_sum_discount_month = min_price_sum_discount_month.rename(columns={"price_sum_discount":"min_price_sum_discount_month"})
    min_price_sum_discount_month["min_price_sum_discount_month"] = min_price_sum_discount_month["min_price_sum_discount_month"].apply(lambda x: log(x + 1, 5))
    max_price_sum_discount_month = pd.DataFrame(min_max_price_sum_discount["price_sum_discount"].groupby([min_max_price_sum_discount["uid"]]).max()).reset_index()
    max_price_sum_discount_month = max_price_sum_discount_month.rename(columns={"price_sum_discount":"max_price_sum_discount_month"})
    max_price_sum_discount_month["max_price_sum_discount_month"] = max_price_sum_discount_month["max_price_sum_discount_month"].apply(lambda x: log(x + 1, 5))


    std_price_sum_discount = history["price_sum_discount"].groupby([history["uid"],history["month"]]).sum().reset_index()
    std_price_sum_discount = std_price_sum_discount.groupby(["uid"]).agg({"price_sum_discount":"std"}).rename(columns={"price_sum":"std_price_sum_discount"}).reset_index()

    #平均折扣
    average_discount = pd.DataFrame({"average_discount":history["discount"].groupby([history["uid"]]).sum()}).reset_index()
    average_discount["average_discount"] = average_discount["average_discount"].apply(lambda x: log(x/NUM + 1, 5))
    #获取每个用户每次购买平均/最大/最小购买量
    average_qty_each = history.groupby(["uid"]).agg({"qty":"mean"}).rename(columns={"qty":"average_qty_each"}).reset_index()
    min_qty = average_qty_each["average_qty_each"].min()
    max_qty = average_qty_each["average_qty_each"].max()
    average_qty_each["average_qty_each"] = average_qty_each["average_qty_each"].apply(lambda x : x-min_qty/(max_qty- min_qty))
    min_qty_each = pd.DataFrame({"min_qty_each" : history["qty"].groupby([history["uid"]]).min()}).reset_index()
    min_qty = min_qty_each["min_qty_each"].min()
    max_qty = min_qty_each["min_qty_each"].max()
    min_qty_each["min_qty_each"] = min_qty_each["min_qty_each"].apply(lambda x : x-min_qty/(max_qty- min_qty))
    max_qty_each = pd.DataFrame({"max_qty_each": history["qty"].groupby([history["uid"]]).max()}).reset_index()
    min_qty = max_qty_each["max_qty_each"].min()
    max_qty = max_qty_each["max_qty_each"].max()
    max_qty_each["max_qty_each"] = max_qty_each["max_qty_each"].apply(lambda x : x-min_qty/(max_qty- min_qty))

    std_qty_each = history.groupby(["uid"]).agg({"qty": "std"}).rename(
        columns={"qty": "std_qty_each"}).reset_index()
    #历史平均折扣力度(discount/总价格)
    average_discount_price_ration = pd.DataFrame({"average_discount_price_ration":history["discount_ratio"].groupby([history["uid"]]).mean()}).reset_index()
    std_discount_price_ration = pd.DataFrame(
        {"std_discount_price_ration": history["discount_ratio"].groupby([history["uid"]]).std()}).reset_index()


    #平均每月免费与非免费的占比
    average_free_no_free_percentage = history.groupby(["uid"]).agg({"free": "sum",
                                                      "uid": "count"}).rename(
        columns={"free": "average_free_percentage", "uid": "uid_count"}).reset_index()
    average_free_no_free_percentage["average_free_percentage"] = average_free_no_free_percentage.apply(lambda x : x["average_free_percentage"] / x["uid_count"] / NUM, axis=1)
    average_free_no_free_percentage["avaerage_no_free_percentage"] = average_free_no_free_percentage.apply(lambda x: 1.0 - x["average_free_percentage"] / NUM, axis=1)
    average_free_no_free_percentage.pop("uid_count")
    #平均/最低/最高单价
    average_single_price = history.groupby(["uid"]).agg({"price":"mean"}).rename(columns={"price":"average_single_price"}).reset_index()
    average_single_price["average_single_price"] = average_single_price[
        "average_single_price"].apply(lambda x: log(x + 1, 5))
    max_single_price = history.groupby(["uid"]).agg({"price":"max"}).rename(columns={"price":"max_single_price"}).reset_index()
    max_single_price["max_single_price"] = max_single_price[
        "max_single_price"].apply(lambda x: log(x + 1, 5))
    min_single_price = history.groupby(["uid"]).agg({"price":"min"}).rename(columns={"price":"min_single_price"}).reset_index()
    min_single_price["min_single_price"] = min_single_price[
        "min_single_price"].apply(lambda x: log(x + 1, 5))

    std_single_price = history.groupby(["uid"]).agg({"price": "std"}).rename(
        columns={"price": "std_single_price"}).reset_index()

    #每月非免费平均总价格/每月非免费平均折扣后的总价/每月非免费平均折扣力度/每月非免费平均折扣
    average_no_free_discount_ratio = pd.DataFrame(
        {"average_no_free_discount_ratio": history.loc[history["free"]!=1]["discount_ratio"].groupby([history["uid"]]).mean()}).reset_index()
    std_no_free_discount_ratio = pd.DataFrame(
        {"std_no_free_discount_ratio": history.loc[history["free"]!=1]["discount_ratio"].groupby([history["uid"]]).std()}).reset_index()

    average_no_free_price_discount_sum_mean = pd.DataFrame({"average_no_free_price_discount_sum_mean": history.loc[history["free"] != 1][
        "price_sum_discount"].groupby([history["uid"]]).mean()}).reset_index()
    average_no_free_price_discount_sum_mean["average_no_free_price_discount_sum_mean"] = average_no_free_price_discount_sum_mean[
        "average_no_free_price_discount_sum_mean"].apply(lambda x: log(x + 1, 5))

    std_no_free_price_discount_sum_mean = pd.DataFrame(
        {"std_no_free_price_discount_sum_mean": history.loc[history["free"] != 1][
            "price_sum_discount"].groupby([history["uid"]]).std()}).reset_index()


    average_no_free_discount_mean = pd.DataFrame({"average_no_free_discount_mean": history.loc[history["free"] != 1][
        "discount"].groupby([history["uid"]]).mean()}).reset_index()
    average_no_free_discount_mean["average_no_free_discount_mean"] = average_no_free_discount_mean[
        "average_no_free_discount_mean"].apply(lambda x: log(x + 1, 5))
    std_no_free_discount_mean = pd.DataFrame({"std_no_free_discount_mean": history.loc[history["free"] != 1][
        "discount"].groupby([history["uid"]]).std()}).reset_index()

    average_no_free_price_sum_mean = pd.DataFrame({"average_no_free_price_sum_mean":history.loc[history["free"]!=1]["price_sum"].groupby([history["uid"]]).mean()}).reset_index()
    average_no_free_price_sum_mean["average_no_free_price_sum_mean"] = average_no_free_price_sum_mean[
        "average_no_free_price_sum_mean"].apply(lambda x: log(x + 1, 5))

    std_no_free_price_sum_mean = pd.DataFrame({"std_no_free_price_sum_mean":history.loc[history["free"]!=1]["price_sum"].groupby([history["uid"]]).std()}).reset_index()

    average_no_free_single_price_mean = pd.DataFrame({"average_no_free_single_price_mean":history.loc[history["free"]!=1]["price"].groupby([history["uid"]]).mean()}).reset_index()
    average_no_free_single_price_mean["average_no_free_single_price_mean"] = average_no_free_single_price_mean[
        "average_no_free_single_price_mean"].apply(lambda x: log(x + 1, 5))

    std_no_free_single_price_mean = pd.DataFrame({"std_no_free_single_price_mean":
                                                          history.loc[history["free"] != 1]["price"].groupby(
                                                              [history["uid"]]).std()}).reset_index()


    #获取每个用户当月的购买物品总价格和
    current = order.loc[order["month"]==MONTH]
    current_price_sum = pd.DataFrame({"current_price_sum":current["price_sum_discount"].groupby([current["uid"]]).sum()}).reset_index()
    current_price_sum["current_price_sum"] = current_price_sum["current_price_sum"].apply(lambda x: log(x + 1, 5))
    current_price_sum_discount_average= pd.DataFrame({"current_price_sum_discount_average":current["price_sum_discount"].groupby([current["uid"]]).mean()}).reset_index()
    current_price_sum_discount_average["current_price_sum_discount_average"] = current_price_sum_discount_average["current_price_sum_discount_average"].apply(lambda x: log(x + 1, 5))
    #获取每个用户当月的总消费次数、免费与非免费的比例
    current_num_order = current.groupby(["uid"]).agg({"free":"sum",
                                                      "uid":"count"}).rename(columns={"free":"current_free_percentage","uid":"current_order_num"}).reset_index()
    current_num_order["current_free_percentage"] = current_num_order.apply(lambda x : x["current_free_percentage"] / x["current_order_num"], axis=1)
    current_num_order["current_no_free_percentage"] = current_num_order.apply(lambda x: 1.0 - x["current_free_percentage"], axis=1)

    current_no_free_price_sum_mean = pd.DataFrame({"current_no_free_price_sum_mean":current.loc[current["free"]!=1]["price_sum"].groupby([current["uid"]]).mean()}).reset_index()
    current_no_free_price_sum_mean["current_no_free_price_sum_mean"] = current_no_free_price_sum_mean[
        "current_no_free_price_sum_mean"].apply(lambda x: log(x + 1, 5))

    current_no_free_price_discount_sum_mean = pd.DataFrame({"current_no_free_price_discount_sum_mean": current.loc[current["free"] != 1][
        "price_sum_discount"].groupby([current["uid"]]).mean()}).reset_index()
    current_no_free_price_discount_sum_mean["current_no_free_price_discount_sum_mean"] = current_no_free_price_discount_sum_mean[
        "current_no_free_price_discount_sum_mean"].apply(lambda x: log(x + 1, 5))

    current_no_free_discount_mean = pd.DataFrame({"current_no_free_discount_mean": current.loc[current["free"] != 1][
        "discount"].groupby([current["uid"]]).mean()}).reset_index()
    current_no_free_discount_mean["current_no_free_discount_mean"] = current_no_free_discount_mean[
        "current_no_free_discount_mean"].apply(lambda x: log(x + 1, 5))

    current_no_free_discount_ratio_mean = pd.DataFrame({"current_no_free_discount_ratio_mean": current.loc[current["free"] != 1][
        "discount_ratio"].groupby([current["uid"]]).mean()}).reset_index()
    #当月平均单价
    current_price_mean = current.groupby(["uid"]).agg({"price":"mean"}).rename(columns={"price":"current_price_mean"}).reset_index()
    current_price_mean["current_price_mean"] = current_price_mean[
        "current_price_mean"].apply(lambda x: log(x + 1, 5))

    std_current_price_mean = current.groupby(["uid"]).agg({"price": "std"}).rename(
        columns={"price": "std_current_price_mean"}).reset_index()
    #当月非免费平均单价
    current_no_freeprice_mean = current.loc[current["free"]!=1].groupby(["uid"]).agg({"price":"mean"}).rename(columns={"price":"current_no_freeprice_mean"}).reset_index()
    current_no_freeprice_mean["current_no_freeprice_mean"] = current_no_freeprice_mean[
        "current_no_freeprice_mean"].apply(lambda x: log(x + 1, 5))
    std_current_no_freeprice_mean = current.loc[current["free"]!=1].groupby(["uid"]).agg({"price":"std"}).rename(columns={"price":"std_current_no_freeprice_mean"}).reset_index()
    #当月每次平均折扣/每次折扣力度
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



    features = [average_price_each, min_price_each, max_price_each, average_discount,average_discount_price_ration,
                average_price,
                max_price_month,
                min_price_month,
                average_num_order,
                min_num_order,
                max_num_order,
                average_qty_each,
                min_qty_each,
                max_qty_each,
                average_free_no_free_percentage,
                current_price_sum,
                current_price_sum_discount_average,
                current_num_order,
                current_no_free_price_sum_mean,
                current_no_free_price_discount_sum_mean,
                current_no_free_discount_mean,
                current_no_free_discount_ratio_mean,
                current_price_mean,
                average_single_price,
                average_no_free_discount_ratio,
                average_no_free_price_discount_sum_mean,
                average_no_free_discount_mean,
                average_no_free_price_sum_mean,
                max_single_price,
                min_single_price,
                average_price_sum_discount,
                min_price_sum_discount_month,
                max_price_sum_discount_month,
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
                std_single_price,
                std_discount_price_ration,
                std_price_sum_discount,
                std_qty_each,
                std_price_sum,
                std_price_each,
                std_num_order,
                price_mean,
                price_sum_mean]

    return features

def get_loan_feature(loan, MONTH, NUM, uid):
    #特征（逐个计算）
    loan["remain_loan"] = loan.loc[loan["month"]<=MONTH].apply(get_remain_loan, axis=1, args=(MONTH,))
    loan["remain_loan"] = loan["remain_loan"] .fillna(0)
    loan["remain_pay"] = loan.loc[loan["month"] <= MONTH].apply(get_remain_pay, axis=1, args=(MONTH,))
    loan["remain_pay"] = loan["remain_pay"].fillna(0)
    loan_split = loan.loc[loan["month"] <= MONTH]
    
	
	
	#统计所有人平均每次贷款的平均额度以及月供的平均值
    static_loan_sum_mean = np.array(loan_split["loan_amount"]).mean()
    static_pay_per_month_mean = np.array(loan_split["pay_per_month"]).mean()
    #######
    tmp_loan_sum_mean = loan_split.groupby(["uid"]).agg({"loan_amount": "mean"}).rename(
        columns={"loan_amount": "loan_sum_mean"}).reset_index()
    loan_sum_mean = pd.DataFrame({"uid":uid})
    loan_sum_mean = pd.merge(loan_sum_mean, tmp_loan_sum_mean, on=["uid"], how="left")
    loan_sum_mean = loan_sum_mean.fillna(0.0)
    loan_sum_mean["loan_sum_mean"] = loan_sum_mean["loan_sum_mean"].apply(lambda x: x - static_loan_sum_mean)
    #tmp_loan_sum_mean_array = np.array(loan_sum_mean["loan_sum_mean"])
    #tmp_loan_sum_mean_array.sort()
    #loan_sum_mean["loan_sum_mean_over"] = loan_sum_mean["loan_sum_mean"].apply(lambda x : get_over(x, tmp_loan_sum_mean_array))
    #####
    tmp_pay_per_month_mean = loan_split.groupby(["uid"]).agg({"pay_per_month": "mean"}).rename(
        columns={"pay_per_month": "pay_per_month_mean"}).reset_index()
    pay_per_month_mean = pd.DataFrame({"uid":uid})
    pay_per_month_mean = pd.merge(pay_per_month_mean, tmp_pay_per_month_mean, on=["uid"], how="left")
    pay_per_month_mean = pay_per_month_mean.fillna(0.0)
    #tmp_pay_per_month_mean_array = np.array(pay_per_month_mean["pay_per_month_mean"])
    #tmp_pay_per_month_mean_array.sort()
    #pay_per_month_mean["pay_per_month_over"] = pay_per_month_mean["pay_per_month_mean"].apply(lambda x : get_over(x, tmp_pay_per_month_mean_array))
    pay_per_month_mean["pay_per_month_mean"] = pay_per_month_mean["pay_per_month_mean"].apply(lambda x : x - static_pay_per_month_mean)

	
	
	
	
	


    #贷款概率
    loan_probability = loan_split.groupby(["uid","month"]).agg({"uid":"count"}).rename(columns={"uid":"count_uid"}).reset_index()
    loan_probability = loan_probability.groupby(["uid"]).agg({"month":"count"}).rename(columns={"month":"loan_probability"}).reset_index()
    loan_probability["loan_probability"] = loan_probability["loan_probability"].apply(lambda x : x/NUM)

    #平均每月贷款次数
    average_num_loan = loan_split.groupby(["uid"]).agg({"uid":"count"}).rename(columns={"uid":"average_num_loan"}).reset_index()
    average_num_loan["average_num_loan"] = average_num_loan["average_num_loan"].apply(lambda x : x/NUM)
    min_max_num_loan = pd.DataFrame(loan_split["loan_amount"].groupby([loan["uid"],loan["month"]]).count()).reset_index()
    min_num_loan = pd.DataFrame(min_max_num_loan["loan_amount"].groupby([min_max_num_loan["uid"]]).min()).rename(columns={"loan_amount":"min_num_loan"}).reset_index()
    max_num_loan = pd.DataFrame(min_max_num_loan["loan_amount"].groupby([min_max_num_loan["uid"]]).max()).rename(columns={"loan_amount": "max_num_loan"}).reset_index()

    std_num_loan = loan_split.groupby(["uid","month"]).agg({"uid":"count"}).rename(columns={"uid":"std_num_loan"}).reset_index()
    std_num_loan= std_num_loan.groupby(["uid"]).agg({"std_num_loan":"std"}).reset_index()

    #连续借款
    dicts = {}
    for i in range(8,MONTH+1):
        min_max_num_loan[str(i)] =  min_max_num_loan["month"].apply(lambda x : 1 if x == i else 0)
        dicts[str(i)] = "count"
    min_max_num_loan = min_max_num_loan.groupby(["uid"]).agg(dicts).reset_index()
    for i in range(8, MONTH+1):
        min_max_num_loan["continous_num_loan"] = min_max_num_loan.apply(count_continuous_loan, month=MONTH, axis=1)
    continous_num_loan = pd.DataFrame(min_max_num_loan[["uid","continous_num_loan"]])
    #平均/最高/最低每一次月供
    average_pay_each = loan_split.groupby(["uid"]).agg({"pay_per_month":"mean"}).rename(columns={"pay_per_month":"average_pay_each"}).reset_index()
    average_pay_each["average_pay_each"] = average_pay_each["average_pay_each"].apply(lambda x: log(x + 1, 5))

    std_pay_each= loan_split.groupby(["uid"]).agg({"pay_per_month":"std"}).rename(columns={"pay_per_month":"std_pay_each"}).reset_index()


    min_pay_each = pd.DataFrame({"min_pay_each" : loan_split["pay_per_month"].groupby([loan_split["uid"]]).min()}).reset_index()
    min_pay_each["min_pay_each"] = min_pay_each["min_pay_each"].apply(lambda x: log(x + 1, 5))
    max_pay_each = pd.DataFrame({"max_pay_each": loan_split["pay_per_month"].groupby([loan_split["uid"]]).max()}).reset_index()
    max_pay_each["max_pay_each"] = max_pay_each["max_pay_each"].apply(lambda x: log(x + 1, 5))

    # 平均/最高/最低每一贷款
    average_loan_each = loan_split.groupby(["uid"]).agg({"loan_amount":"mean"}).rename(columns={"loan_amount":"average_loan_each"}).reset_index()
    average_loan_each["average_loan_each"] = average_loan_each["average_loan_each"].apply(lambda x: log(x + 1, 5))

    min_loan_each = pd.DataFrame({"min_loan_each" : loan_split["loan_amount"].groupby([loan_split["uid"]]).min()}).reset_index()
    min_loan_each["min_loan_each"] = min_loan_each["min_loan_each"].apply(lambda x: log(x + 1, 5))

    max_loan_each = pd.DataFrame({"max_loan_each": loan_split["loan_amount"].groupby([loan_split["uid"]]).max()}).reset_index()
    max_loan_each["max_loan_each"] = max_loan_each["max_loan_each"].apply(lambda x: log(x + 1, 5))

    std_loan_each = loan_split.groupby(["uid"]).agg({"loan_amount": "std"}).rename(columns={"loan_amount": "std_loan_each"}).reset_index()

    #平均/最低/最高每月贷款
    average_loan =  pd.DataFrame({"average_loan" : loan_split["loan_amount"].groupby([loan_split["uid"]]).sum()}).reset_index()
    average_loan["average_loan"] = average_loan["average_loan"].apply(lambda x : log(x/NUM+1, 5))
    min_max_loan = pd.DataFrame(loan_split["loan_amount"].groupby([loan_split["uid"],loan["month"]]).sum()).reset_index()
    min_loan_month = pd.DataFrame(min_max_loan["loan_amount"].groupby([min_max_loan["uid"]]).min()).reset_index()
    min_loan_month = min_loan_month.rename(columns={"loan_amount":"min_loan_month"})
    min_loan_month["min_loan_month"] = min_loan_month["min_loan_month"].apply(lambda x: log(x + 1, 5))
    max_loan_month = pd.DataFrame(min_max_loan["loan_amount"].groupby([min_max_loan["uid"]]).max()).reset_index()
    max_loan_month = max_loan_month.rename(columns={"loan_amount": "max_loan_month"})
    max_loan_month["max_loan_month"] = max_loan_month["max_loan_month"].apply(lambda x: log(x + 1, 5))

    std_average_loan = pd.DataFrame(
        {"std_average_loan": loan_split["loan_amount"].groupby([loan_split["uid"], loan_split["month"]]).sum()}).reset_index()
    std_average_loan = std_average_loan.groupby(["uid"]).agg({"std_average_loan":"std"}).reset_index()

    #平均/最低/最高每月月供
    average_pay = pd.DataFrame({"average_pay" : loan_split["pay_per_month"].groupby([loan_split["uid"]]).sum()}).reset_index()
    average_pay["average_pay"] = average_pay["average_pay"].apply(lambda x: log(x/NUM + 1, 5))

    min_max_pay = pd.DataFrame(loan_split["pay_per_month"].groupby([loan_split["uid"],loan["month"]]).sum()).reset_index()

    min_loan_pay = pd.DataFrame(min_max_pay["pay_per_month"].groupby([min_max_pay["uid"]]).min()).reset_index()
    min_loan_pay = min_loan_pay.rename(columns={"pay_per_month":"min_loan_pay"})
    min_loan_pay["min_loan_pay"] = min_loan_pay["min_loan_pay"].apply(lambda x: log(x + 1, 5))

    max_loan_pay = pd.DataFrame(min_max_pay["pay_per_month"].groupby([min_max_pay["uid"]]).max()).reset_index()
    max_loan_pay = max_loan_pay.rename(columns={"pay_per_month":"max_loan_pay"})
    max_loan_pay["max_loan_pay"] = max_loan_pay["max_loan_pay"].apply(lambda x: log(x + 1, 5))


    std_average_pay = pd.DataFrame(
        {"std_average_pay": loan_split["pay_per_month"].groupby([loan_split["uid"], loan_split["month"]]).sum()}).reset_index()
    std_average_pay = std_average_pay.groupby(["uid"]).agg({"std_average_pay":"std"}).reset_index()

    #历史贷款总额
    remain_loan = pd.DataFrame(loan_split["remain_loan"].groupby([loan_split["uid"]]).sum()).reset_index()
    remain_loan["remain_loan"] = remain_loan["remain_loan"].apply(lambda x: log(x + 1, 5))
    #累计月供
    remain_pay = pd.DataFrame(loan_split["remain_pay"].groupby([loan_split["uid"]]).sum()).reset_index()
    remain_pay["remain_pay"] = remain_pay["remain_pay"].apply(lambda x: log(x + 1, 5))
    #平均贷款还款周期
    average_plannum = loan_split.groupby(["uid"]).agg({"plannum": "mean"}).rename(
       columns={"plannum": "average_plannum"}).reset_index()
    average_plannum["average_plannum"] = average_plannum["average_plannum"].apply(lambda x : 12 - x)
    std_plannum = loan_split.groupby(["uid"]).agg({"plannum": "std"}).rename(
       columns={"plannum": "std_plannum"}).reset_index()

    #下个月总计剩余贷款和下个月总月供
    loan["next_month_remain_loan"] = loan.loc[loan["month"]<=MONTH+1].apply(get_remain_loan, axis=1, args=(MONTH+1,))
    loan["next_month_remain_loan"] = loan["next_month_remain_loan"] .fillna(0)
    next_month_remain_loan = pd.DataFrame(loan.loc[loan["month"]<=MONTH]["next_month_remain_loan"].groupby([loan["uid"]]).sum()).reset_index()
    next_month_remain_loan["next_month_remain_loan"] = next_month_remain_loan["next_month_remain_loan"].apply(lambda x: log(x + 1, 5))

    loan["next_month_remain_pay"] = loan.loc[loan["month"]<=MONTH+1].apply(get_remain_pay, axis=1, args=(MONTH+1,))
    loan["next_month_remain_pay"] = loan["next_month_remain_pay"] .fillna(0)
    next_month_remain_pay = pd.DataFrame(loan.loc[loan["month"]<=MONTH]["next_month_remain_pay"].groupby([loan["uid"]]).sum()).reset_index()
    next_month_remain_pay["next_month_remain_pay"] = next_month_remain_pay["next_month_remain_pay"].apply(lambda x: log(x + 1, 5))

    # 到下个月的天数
    loan_split['remain_days'] = loan_split['loan_time'].apply(lambda x: (dt.datetime(2016, MONTH + 1, 1) - parse(x)).days)
    remain_days_mean = pd.DataFrame({"remain_days_mean" : loan_split.groupby('uid')['remain_days'].mean()}).reset_index()
    remain_days_max = pd.DataFrame({"remain_days_max" : loan_split.groupby('uid')['remain_days'].max()}).reset_index()
    remain_days_min = pd.DataFrame({"remain_days_min" : loan_split.groupby('uid')['remain_days'].min()}).reset_index()
    # 计划时间是否超出时间
    loan_split['over'] = loan_split.apply(lambda x: 1 if x['month'] + x['plannum'] <= MONTH else 0, axis=1)
    over_rate = loan_split.groupby(["uid"]).agg({"over":"mean"}).rename( columns={"over": "over_ratio"}).reset_index()


    #当月月供和当月贷款总额
    current_loan = loan.loc[loan["month"] == MONTH]

    current_loan_sum = pd.DataFrame({"current_loan_sum" : current_loan["loan_amount"].groupby([current_loan["uid"]]).sum()}).reset_index()
    current_loan_sum["current_loan_sum"] = current_loan_sum["current_loan_sum"].apply(lambda x: log(x + 1, 5))
    current_pay_sum = pd.DataFrame({"current_pay_sum" : current_loan["pay_per_month"].groupby([current_loan["uid"]]).sum()}).reset_index()
    current_pay_sum["current_pay_sum"] = current_pay_sum["current_pay_sum"].apply(lambda x: log(x + 1, 5))


    current_loan_average = pd.DataFrame({"current_loan_average" : current_loan["loan_amount"].groupby([current_loan["uid"]]).mean()}).reset_index()
    current_loan_average["current_loan_average"] = current_loan_average["current_loan_average"].apply(lambda x: log(x + 1, 5))
    current_pay_average = pd.DataFrame({"current_pay_average" : current_loan["pay_per_month"].groupby([current_loan["uid"]]).mean()}).reset_index()
    current_pay_average["current_pay_average"] = current_pay_average["current_pay_average"].apply(lambda x: log(x + 1, 5))
    current_num_loan = current_loan.groupby(["uid"]).agg({"uid": "count"}).rename(
        columns={"uid": "current_num_loan"}).reset_index()
    current_plannum_mean = current_loan.groupby(["uid"]).agg({"plannum": "mean"}).rename(
        columns={"plannum": "current_plannum_mean"}).reset_index()
    current_plannum_mean["current_plannum_mean"] = current_plannum_mean["current_plannum_mean"].apply(lambda x: 12 - x)

    features = [average_loan, average_pay, remain_loan, remain_pay, average_loan_each, min_loan_each, max_loan_each, average_pay_each, min_pay_each, max_pay_each,
                average_plannum,
                min_loan_pay,
                max_loan_pay,
                max_loan_month,
                min_loan_month,
                next_month_remain_loan,
                next_month_remain_pay,
                remain_days_mean,
                remain_days_max,
                remain_days_min,
                over_rate,
                average_num_loan,
                max_num_loan,
                min_num_loan,
                continous_num_loan,
                current_loan_sum,
                current_pay_sum,
                current_loan_average,
                current_pay_average,
                current_num_loan,
                current_plannum_mean,
                std_pay_each,
                std_loan_each,
                std_average_loan,
                std_plannum,
                std_average_pay,
                std_num_loan,
                loan_probability,
                loan_sum_mean,
                pay_per_month_mean]
    return features

def get_click_feature(click, MONTH):
    click_current_month = click.loc[click["month"]==MONTH].groupby(["uid"]).agg({"pid_1":"sum",
                                                 "pid_2":"sum",
                                                 "pid_3":"sum",
                                                 "pid_4":"sum",
                                                 "pid_5":"sum",
                                                 "pid_6":"sum",
                                                 "pid_7":"sum",
                                                 "pid_8":"sum",
                                                 "pid_9":"sum",
                                                 "pid_10":"sum",
                                                  "pid":"count"}).rename( columns={"pid": "pid_count"}).reset_index()
    click_current_month["pid_1"] = click_current_month.apply(lambda x: x["pid_1"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_2"] = click_current_month.apply(lambda x: x["pid_2"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_3"] = click_current_month.apply(lambda x: x["pid_3"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_4"] = click_current_month.apply(lambda x: x["pid_4"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_5"] = click_current_month.apply(lambda x: x["pid_5"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_6"] = click_current_month.apply(lambda x: x["pid_6"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_7"] = click_current_month.apply(lambda x: x["pid_7"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_8"] = click_current_month.apply(lambda x: x["pid_8"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_9"] = click_current_month.apply(lambda x: x["pid_9"] / x["pid_count"] * 100, axis=1)
    click_current_month["pid_10"] = click_current_month.apply(lambda x: x["pid_10"] / x["pid_count"] * 100, axis=1)
    click_current_month.pop("pid_count")
    return [click_current_month]


#新的获取7/15/30的统计数据

def is_sampled_data(x,start,end, strip_format='%Y-%m-%d'):
    tmp = datetime.datetime.strptime(x, strip_format)
    if (tmp - start).days >=0 and (end-tmp).days > 0:
        return 1
    return 0

	##################################################占比
def capture_loan_information(raw_feature, gap, end_month):
    feature = cp.deepcopy(raw_feature)
    end_time = datetime.datetime.strptime('2016-{}-01'.format(end_month), '%Y-%m-%d')
    if gap < 30:
        delta = datetime.timedelta(days=gap)
        start_time = end_time - delta
    elif gap == 30:
        start_time = datetime.datetime.strptime('2016-{}-01'.format(end_month-1), '%Y-%m-%d')
    elif gap == 60:
        start_time = datetime.datetime.strptime('2016-{}-01'.format(end_month - 2), '%Y-%m-%d')

    feature["sampled"] = feature["loan_time"].apply(is_sampled_data, start=start_time, end= end_time)
    splited_feature = feature.loc[feature["sampled"]==1]
    feature.pop("sampled")
    splited_feature.pop("sampled")
    #提取特征
    columns_name = ["loan_amount","plannum","pay_per_month","uid"]
    actions = ["mean","mean","mean","count"]
    agg_dict = {}
    rename_dict = {}
    money_list = []
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = name + "_"+ action + "_" + str(gap)
        if name != "uid" and name != "plannum":
            money_list.append(rename_dict[name])

    feature_loan_0 = splited_feature.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()
    for name in money_list:
        feature_loan_0[name] = feature_loan_0[name].apply(lambda x : log(x+1, 5))
    feature_loan_0["plannum_mean_{}".format(gap)] = feature_loan_0["plannum_mean_{}".format(gap)].apply(lambda x: 12 - x)

    columns_name = ["loan_amount","pay_per_month"]
    actions = ["sum","sum"]
    agg_dict = {}
    rename_dict = {}
    money_list = []
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = name + "_"+ action + "_" + str(gap)
        money_list.append(rename_dict[name])
    feature_loan_1 = splited_feature.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()


    current = feature.loc[feature["month"]== end_month-1].groupby(["uid"]).agg({"loan_amount":"sum","pay_per_month":"sum"}).rename(columns={
        "loan_amount":"loan_amount_current",
         "pay_per_month":"pay_per_month_current"
    }).reset_index()

    feature_loan_1 = pd.merge(current, feature_loan_1, on=["uid"], how="left")
    feature_loan_1 = feature_loan_1.fillna(0.0)
    feature_loan_1["loan_amount_current"] = feature_loan_1.apply(lambda x : x["loan_amount_sum_{}".format(gap)] / x["loan_amount_current"] , axis=1)
    feature_loan_1["pay_per_month_current"] = feature_loan_1.apply(lambda x : x["pay_per_month_sum_{}".format(gap)] / x["loan_amount_current"] , axis=1)


    for name in money_list:
        feature_loan_1[name] = feature_loan_1[name].apply(lambda x: log(x + 1, 5))

    columns_name = ["loan_amount","pay_per_month"]
    actions = ["std","std"]
    agg_dict = {}
    rename_dict = {}
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = name + "_"+ action + "_" + str(gap)
    feature_loan_2 = splited_feature.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()

    return [feature_loan_0,feature_loan_1, feature_loan_2]

def capture_order_information(raw_feature, gap, end_month):
    feature = cp.deepcopy(raw_feature)
    end_time = datetime.datetime.strptime('2016-{}-01'.format(end_month), '%Y-%m-%d')
    if gap < 30:
        delta = datetime.timedelta(days=gap)
        start_time = end_time - delta
    elif gap == 30:
        start_time = datetime.datetime.strptime('2016-{}-01'.format(end_month-1), '%Y-%m-%d')
    elif gap == 60:
        start_time = datetime.datetime.strptime('2016-{}-01'.format(end_month - 2), '%Y-%m-%d')

    feature["sampled"] = feature["buy_time"].apply(is_sampled_data, start=start_time, end= end_time, strip_format="%Y-%m-%d")
    splited_feature = feature.loc[feature["sampled"]==1]
    feature.pop("sampled")
    splited_feature.pop("sampled")

    columns_name = ["price","discount_ratio","discount","price_sum","qty","price_sum_discount","uid"]
    actions = ["mean","mean","mean","mean", "mean", "mean","count"]
    agg_dict = {}
    rename_dict = {}
    money_list = []
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = "total" + name + "_"+ action + "_" + str(gap)
        if name != "uid" and name != "qty":
            money_list.append(rename_dict[name])
    total = splited_feature.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()
    for name in money_list:
        total[name] = total[name].apply(lambda x : log(x+1, 5))

    columns_name = ["price","discount_ratio","discount","price_sum","qty","price_sum_discount"]
    actions = ["std","std","std","std", "std", "std"]
    agg_dict = {}
    rename_dict = {}
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = "total" + name + "_"+ action + "_" + str(gap)
    total_1 = splited_feature.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()


    #
    no_free = splited_feature.loc[splited_feature["free"]!=1]
    free = splited_feature.loc[splited_feature["free"]==1]
    #
    columns_name = ["price","discount_ratio","discount","price_sum","qty","price_sum_discount","uid"]
    actions = ["mean","mean","mean","mean", "mean", "mean","count"]
    agg_dict = {}
    rename_dict = {}
    money_list = []
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = "no_free_" + name + "_"+ action + "_" + str(gap)
        if name != "uid" and name != "qty":
            money_list.append(rename_dict[name])
    no_free_0 = no_free.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()
    for name in money_list:
        no_free_0[name] = no_free_0[name].apply(lambda x : log(x+1, 5))

    min_qty = no_free_0["no_free_qty_mean_{}".format(gap)].min()
    max_qty = no_free_0["no_free_qty_mean_{}".format(gap)].max()
    no_free_0["no_free_qty_mean_{}".format(gap)] = no_free_0["no_free_qty_mean_{}".format(gap)].apply(lambda x : x-min_qty/(max_qty- min_qty))
    #
    columns_name = ["price","discount_ratio","discount","price_sum","qty","price_sum_discount"]
    actions = ["std","std","std","std", "std", "std"]
    agg_dict = {}
    rename_dict = {}
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = "no_free_" + name + "_"+ action + "_" + str(gap)
    no_free_1 = no_free.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()



    columns_name = ["uid","qty"]
    actions = ["count","mean"]
    agg_dict = {}
    rename_dict = {}
    for name, action in zip(columns_name, actions):
        agg_dict[name] = action
        rename_dict[name] = "free_" + name + "_"+ action + "_" + str(gap)
    free = free.groupby(["uid"]).agg(agg_dict).rename(columns=rename_dict).reset_index()
    min_qty = free["free_qty_mean_{}".format(gap)].min()
    max_qty = free["free_qty_mean_{}".format(gap)].max()
    free["free_qty_mean_{}".format(gap)] = free["free_qty_mean_{}".format(gap)].apply(lambda x : x-min_qty/(max_qty- min_qty))

    total_0 = pd.merge(no_free_0, free, on=["uid"], how="left")
    total_0 = pd.merge(total_0, no_free_1, on=["uid"], how="left")
    total_0 = total_0.fillna(0)

    total_0["num_order"] = total_0.apply(lambda x : x["free_uid_count_{}".format(gap)] + x["no_free_uid_count_{}".format(gap)], axis=1 )
    total_0["free_uid_count_{}".format(gap)] = total_0.apply(lambda x : x["free_uid_count_{}".format(gap)] / x["num_order"], axis=1)
    total_0["no_free_uid_count_{}".format(gap)] = total_0.apply(lambda x: x["no_free_uid_count_{}".format(gap)] / x["num_order"], axis=1)

    total_0 = total_0.rename(columns={"free_uid_count_{}".format(gap):"free_uid_ratio_{}".format(gap),
                                  "no_free_uid_count_{}".format(gap): "no_free_uid_ratio_{}".format(gap)})
    #


    average_order_sum = splited_feature.groupby(["uid"]).agg({"price_sum_discount":"mean"}).rename(columns={"price_sum_discount":"price_sum_discount_mean_{}".format(gap)}).reset_index()
    average_order_sum["price_sum_discount_mean_{}".format(gap)] = average_order_sum["price_sum_discount_mean_{}".format(gap)].apply(lambda x : log(x+1,5))



    discount_sum = splited_feature.groupby(["uid"]).agg({"discount":"sum"}).rename(columns={"discount":"discount_sum_{}".format(gap)}).reset_index()
    discount_sum["discount_sum_{}".format(gap)] = discount_sum["discount_sum_{}".format(gap)].apply(lambda x : log(x+1,5))

    average_discount_sum = splited_feature.groupby(["uid"]).agg({"discount":"mean"}).rename(columns={"discount":"discount_mean{}".format(gap)}).reset_index()
    average_discount_sum["discount_mean{}".format(gap)] = average_discount_sum["discount_mean{}".format(gap)].apply(lambda x : log(x+1,5))


    #
    order_sum = splited_feature.groupby(["uid"]).agg({"price_sum_discount":"sum"}).rename(columns={"price_sum_discount":"price_sum_discount_sum_{}".format(gap)}).reset_index()
    current = feature.loc[feature["month"]==end_month-1].groupby(["uid"]).agg({"price_sum_discount":"sum"}).rename(columns={"price_sum_discount":"price_sum_discount_current_month"}).reset_index()
    order_sum = pd.merge(current, order_sum, on=["uid"], how="left")
    order_sum = order_sum.fillna(0.0)
    order_sum["price_sum_discount_current_month"] = order_sum.apply(lambda x : x["price_sum_discount_sum_{}".format(gap)] / x["price_sum_discount_current_month"], axis=1)
    order_sum["price_sum_discount_sum_{}".format(gap)] = order_sum["price_sum_discount_sum_{}".format(gap)].apply(lambda x : log(x+1,5))


    return [total, total_0, total_1, order_sum, average_order_sum, discount_sum, average_discount_sum]





def capture_order_loan_cross_information(row_order, row_loan, gap, end_month):
    order = cp.deepcopy(row_order)
    loan = cp.deepcopy(row_loan)

    end_time = datetime.datetime.strptime('2016/{}/01'.format(end_month), '%Y/%m/%d')
    if gap < 30:
        delta = datetime.timedelta(days=gap)
        start_time = end_time - delta
    elif gap == 30:
        start_time = datetime.datetime.strptime('2016/{}/01'.format(end_month-1), '%Y/%m/%d')
    elif gap == 60:
        start_time = datetime.datetime.strptime('2016/{}/01'.format(end_month - 2), '%Y/%m/%d')

    order["sampled"] = order["buy_time"].apply(is_sampled_data, start=start_time, end=end_time,
                                                   strip_format="%Y-%m-%d")
    loan["sampled"] = loan["loan_time"].apply(is_sampled_data, start=start_time, end=end_time)


    splited_order = order.loc[order["sampled"]==1]
    splited_loan = loan.loc[loan["sampled"]==1]

    splited_order.pop("sampled")
    splited_loan.pop("sampled")

    loan.pop("sampled")
    order.pop("sampled")

    splited_loan = splited_loan.groupby(["uid"]).agg({"loan_amount":"sum",
                                                      "uid":"count"}).rename(columns={"loan_amount":"loan_amount_sum",
                                                                                      "uid":"loan_num"}).reset_index()

    splited_order = splited_order.loc[splited_order["free"]!=1]
    splited_order = splited_order.groupby(["uid"]).agg({"price_sum_discount":"sum",
                                                        "uid":"count"}).rename(columns={"price_sum_discount":"price_sum_discount",
                                                                                        "uid": "order_num"
                                                                                        }).reset_index()
    total = pd.merge(splited_order,splited_loan, on="uid", how="left")
    total = total.fillna(0.0)

    total["loan_and_order_{}".format(gap)] = total.apply(lambda x: 1 if x["loan_amount_sum"] > 0 and x["price_sum_discount"] > 0 else 0, axis=1)
    total["has_order_{}".format(gap)] = total.apply(
        lambda x: 1 if x["price_sum_discount"] > 0 else 0, axis=1)
    total["has_loan_{}".format(gap)] = total.apply(
        lambda x: 1 if x["loan_amount_sum"] > 0 else 0, axis=1)
    total["no_loan_and_no_order_{}".format(gap)] = total.apply(
        lambda x: 1 if x["loan_amount_sum"] == 0 and x["price_sum_discount"] == 0 else 0, axis=1)


    total["loan_price_ration_{}".format(gap)] = total.apply(lambda x : x["loan_amount_sum"] /  x["price_sum_discount"] if x["has_order_{}".format(gap)] == 1 else 0, axis=1)
    total["loan_price_num_ration_{}".format(gap)] = total.apply(
        lambda x: x["loan_num"] / x["order_num"] if x["has_order_{}".format(gap)] == 1 else 0, axis=1)
    total = pd.DataFrame(total[["uid",
                                "loan_price_ration_{}".format(gap),
                                "loan_and_order_{}".format(gap),
                                "has_order_{}".format(gap),
                                "has_loan_{}".format(gap),
                                "no_loan_and_no_order_{}".format(gap)]])
    return [total]

def capture_user_information(feature):
    cp_feature = cp.deepcopy(feature)
    cp_feature.pop("active_date")
    return cp_feature

def get_user_loan_feature(loan,MONTH, start_month=8):
    user = pd.read_csv("../data/t_user.csv")
    user["limit"] = user["limit"].apply(lambda x : 5**x-1)

    # 提取历史贷款信息
    loan["month"] = loan["loan_time"].apply(split_by_month)
    # 分割
    loan = loan.loc[(loan["month"] >= start_month) & (loan["month"] <= MONTH)]
    loan_amount_sum = pd.DataFrame({"loan_amount_sum": loan["loan_amount"].groupby([loan["uid"]]).sum()}).reset_index()

    # 贷款总额和初始的差
    user_loan = pd.merge(user, loan_amount_sum, how='left', on='uid')
    user_loan.fillna(0)
    # user_loan['diff_loan'] = user_loan.apply(lambda x: x['limit'] - x['loan_amount_sum'], axis=1)
    user_loan['diff_loan'] = user_loan.apply(lambda x: 1 if x['limit'] - x['loan_amount_sum'] else 0, axis=1)
    # user_loan["diff_loan"] = user_loan["diff_loan"].apply(lambda x: log(x + 1, 5))

    user_loan = user_loan.loc[:, ['uid', 'diff_loan']]
    return user_loan