from sys import path
path.append('.')
from tl.src.util import *
from math import log
import datetime as dt

def get_loan_feature(MONTH, NUM, uid, loan):

    # 提取历史贷款信息
    loan["month"] = loan["loan_time"].apply(split_by_month)
    # 特征（逐个计算）
    loan["pay_per_month"] = loan.apply(get_pay_per_month, axis=1)
    loan["remain_loan"] = loan.loc[loan["month"] <= MONTH].apply(get_remain_loan, axis=1, args=(MONTH,))
    loan["remain_loan"] = loan["remain_loan"].fillna(0)
    loan["remain_pay"] = loan.loc[loan["month"] <= MONTH].apply(get_remain_pay, axis=1, args=(MONTH,))
    loan["remain_pay"] = loan["remain_pay"].fillna(0)
    # 汇总

    # 分割
    loan_split = loan.loc[loan["month"] <= MONTH]

    # 到下个月的天数
    loan_split['remain_days'] = loan_split['loan_time'].apply(lambda x: (dt.datetime(2016, MONTH + 1, 1) - parse(x)).days)
    remain_days_mean = pd.DataFrame(loan_split.groupby('uid')['remain_days'].mean()).reset_index()
    remain_days_max = pd.DataFrame(loan_split.groupby('uid')['remain_days'].max()).reset_index()
    remain_days_min = pd.DataFrame(loan_split.groupby('uid')['remain_days'].min()).reset_index()

    # amount_sum = pd.DataFrame(loan_split.groupby('uid')['loan_amount'].sum()).reset_index()
    # remain_days_sum = loan_split.groupby('uid')['remain_days'].sum()
    # loan_split['remain_per_day'] = remain_days_sum / amount_sum

    # 计划时间是否超出时间
    loan_split['over'] = loan_split.apply(lambda x: 1 if x['month'] + x['plannum'] <= MONTH else 0, axis=1)
    over_rate = pd.DataFrame(loan_split.groupby('uid')['over'].sum() / loan_split.groupby('uid').size()).reset_index()


    # 平均每月贷款
    average_loan = pd.DataFrame({"average_loan": loan.loc[loan["month"] <= MONTH]["loan_amount"].groupby([loan["uid"]]).sum()}).reset_index()
    average_loan["average_loan"] = average_loan["average_loan"].apply(lambda x: log(x / NUM + 1, 5))

    # 平均/最低/最高每月月供
    average_pay = pd.DataFrame({"average_pay": loan.loc[loan["month"] <= MONTH]["pay_per_month"].groupby([loan["uid"]]).sum()}).reset_index()
    average_pay["average_pay"] = average_pay["average_pay"].apply(lambda x: log(x / NUM + 1, 5))

    # min_max_pay = pd.DataFrame(loan.loc[loan["month"] <= MONTH]["pay_per_month"].groupby([loan["uid"], loan["month"]]).sum()).reset_index()

    # min_loan_pay = pd.DataFrame(min_max_pay["pay_per_month"].groupby([loan["uid"]]).min()).reset_index()
    # min_loan_pay = min_loan_pay.rename(columns={"pay_per_month": "min_loan_pay"})
    # min_loan_pay["min_loan_pay"] = min_loan_pay["min_loan_pay"].apply(lambda x: log(x + 1, 5))

    # max_loan_pay = pd.DataFrame(min_max_pay["pay_per_month"].groupby([loan["uid"]]).max()).reset_index()
    # max_loan_pay = max_loan_pay.rename(columns={"pay_per_month": "max_loan_pay"})
    # max_loan_pay["max_loan_pay"] = max_loan_pay["max_loan_pay"].apply(lambda x: log(x + 1, 5))

    # 历史贷款总额
    remain_loan = pd.DataFrame(loan.loc[loan["month"] <= MONTH]["remain_loan"].groupby([loan["uid"]]).sum()).reset_index()
    remain_loan["remain_loan"] = remain_loan["remain_loan"].apply(lambda x: log(x + 1, 5))

    # 累计月供
    remain_pay = pd.DataFrame(loan.loc[loan["month"] <= MONTH]["remain_pay"].groupby([loan["uid"]]).sum()).reset_index()
    remain_pay["remain_pay"] = remain_pay["remain_pay"].apply(lambda x: log(x + 1, 5))

    # 当月月供和当月贷款总额
    current_loan_sum = pd.DataFrame({"current_loan_sum": loan.loc[loan["month"] == MONTH]["loan_amount"].groupby([loan["uid"]]).sum()}).reset_index()
    current_loan_sum["current_loan_sum"] = current_loan_sum["current_loan_sum"].apply(lambda x: log(x + 1, 5))
    current_pay_sum = pd.DataFrame({"current_pay_sum": loan.loc[loan["month"] == MONTH]["pay_per_month"].groupby([loan["uid"]]).sum()}).reset_index()
    current_pay_sum["current_pay_sum"] = current_pay_sum["current_pay_sum"].apply(lambda x: log(x + 1, 5))

    # 平均/最高/最低每一次月供
    average_pay_each = loan.loc[loan["month"] <= MONTH].groupby(["uid"]).agg({"pay_per_month": "mean"}).rename(
        columns={"pay_per_month": "average_pay_each"}).reset_index()
    average_pay_each["average_pay_each"] = average_pay_each["average_pay_each"].apply(lambda x: log(x + 1, 5))
    # min_pay_each = pd.DataFrame(
    #     {"min_pay_each": loan.loc[loan["month"] <= MONTH]["pay_per_month"].groupby([loan["uid"]]).min()}).reset_index()
    # min_pay_each["min_pay_each"] = min_pay_each["min_pay_each"].apply(lambda x: log(x + 1, 5))
    # max_pay_each = pd.DataFrame(
    #     {"max_pay_each": loan.loc[loan["month"] <= MONTH]["pay_per_month"].groupby([loan["uid"]]).max()}).reset_index()
    # max_pay_each["max_pay_each"] = max_pay_each["max_pay_each"].apply(lambda x: log(x + 1, 5))

    # 平均/最高/最低每一贷款
    average_loan_each = loan.loc[loan["month"] <= MONTH].groupby(["uid"]).agg({"loan_amount": "mean"}).rename(
        columns={"loan_amount": "average_loan_each"}).reset_index()
    average_loan_each["average_loan_each"] = average_loan_each["average_loan_each"].apply(lambda x: log(x + 1, 5))
    # min_loan_each = pd.DataFrame(
    #     {"min_loan_each": loan.loc[loan["month"] <= MONTH]["loan_amount"].groupby([loan["uid"]]).min()}).reset_index()
    # min_loan_each["min_loan_each"] = min_loan_each["min_loan_each"].apply(lambda x: log(x + 1, 5))
    # max_loan_each = pd.DataFrame(
    #     {"max_loan_each": loan.loc[loan["month"] <= MONTH]["loan_amount"].groupby([loan["uid"]]).max()}).reset_index()
    # max_loan_each["max_loan_each"] = max_loan_each["max_loan_each"].apply(lambda x: log(x + 1, 5))

    feature_loan = pd.merge(uid, average_loan, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, average_pay, on=["uid"], how="left")
    # feature_loan = pd.merge(feature_loan, min_loan_pay, on=["uid"], how="left")
    # feature_loan = pd.merge(feature_loan, max_loan_pay, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, remain_loan, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, remain_pay, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, current_pay_sum, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, current_loan_sum, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, average_pay_each, on=["uid"], how="left")
    # feature_loan = pd.merge(feature_loan, min_pay_each, on=["uid"], how="left")
    # feature_loan = pd.merge(feature_loan, max_pay_each, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, average_loan_each, on=["uid"], how="left")
    # feature_loan = pd.merge(feature_loan, min_loan_each, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, remain_days_mean, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, remain_days_max, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, remain_days_min, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, over_rate, on=["uid"], how="left")
    # feature_loan = pd.merge(feature_loan, amount_sum, on=["uid"], how="left")

    return feature_loan
