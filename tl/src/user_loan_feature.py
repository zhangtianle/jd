from sys import path

path.append('../../')
from tl.src.util import split_by_month
import pandas as pd


def get_user_loan_feature(user, loan, start_month, MONTH):
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
