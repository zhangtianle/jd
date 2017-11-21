from sys import path
path.append('.')
from tl.src.util import *
from math import log


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
    # 平均每月贷款
    average_loan = pd.DataFrame(
        {"average_loan": loan.loc[loan["month"] <= MONTH]["loan_amount"].groupby([loan["uid"]]).sum()}).reset_index()
    average_loan["average_loan"] = average_loan["average_loan"].apply(lambda x: log(x / NUM + 1, 5))
    # 平均每月月供
    average_pay = pd.DataFrame(
        {"average_pay": loan.loc[loan["month"] <= MONTH]["pay_per_month"].groupby([loan["uid"]]).sum()}).reset_index()
    average_pay["average_pay"] = average_pay["average_pay"].apply(lambda x: log(x / NUM + 1, 5))
    # 历史贷款总额
    remain_loan = pd.DataFrame(
        loan.loc[loan["month"] <= MONTH]["remain_loan"].groupby([loan["uid"]]).sum()).reset_index()
    remain_loan["remain_loan"] = remain_loan["remain_loan"].apply(lambda x: log(x + 1, 5))
    # 累计月供
    remain_pay = pd.DataFrame(loan.loc[loan["month"] <= MONTH]["remain_pay"].groupby([loan["uid"]]).sum()).reset_index()
    remain_pay["remain_pay"] = remain_pay["remain_pay"].apply(lambda x: log(x + 1, 5))
    # 当月月供和当月贷款总额
    current_loan_sum = pd.DataFrame(
        {"current_loan_sum": loan.loc[loan["month"] == MONTH]["loan_amount"].groupby(
            [loan["uid"]]).sum()}).reset_index()
    current_loan_sum["current_loan_sum"] = current_loan_sum["current_loan_sum"].apply(lambda x: log(x + 1, 5))
    current_pay_sum = pd.DataFrame(
        {"current_pay_sum": loan.loc[loan["month"] == MONTH]["pay_per_month"].groupby(
            [loan["uid"]]).sum()}).reset_index()
    current_pay_sum["current_pay_sum"] = current_pay_sum["current_pay_sum"].apply(lambda x: log(x + 1, 5))

    feature_loan = pd.merge(uid, average_loan, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, average_pay, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, remain_loan, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, remain_pay, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, current_pay_sum, on=["uid"], how="left")
    feature_loan = pd.merge(feature_loan, current_loan_sum, on=["uid"], how="left")

    return feature_loan
