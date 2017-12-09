from sys import path
path.append('../../')
from tl.src.util import get_url, read_data, split_by_month, change_loan
from math import log
import pandas as pd

root_dir, train_url, feature_url = get_url()
loan, user, order, click = read_data()

loan = loan.fillna(0)
# 提取历史贷款信息
loan["month"] = loan["loan_time"].apply(split_by_month)

# 分割
loan = loan[loan["month"] == 11]

loan['loan_amount'] = change_loan(loan['loan_amount'])
loan_amount = pd.DataFrame({"loan_sum": loan.groupby(loan['uid'])['loan_amount'].sum()}).reset_index()

loan_amount["loan_sum"] = loan_amount['loan_sum'].apply(lambda x: log(x + 1, 5))

loan_amount.to_csv(feature_url + "loan_sum_11.csv")

loan_next_month = pd.merge(pd.DataFrame(user['uid']), loan_amount, on=["uid"], how="left")
loan_next_month["loan_sum"] = loan_next_month["loan_sum"].fillna(0.0)
loan_next_month.to_csv(feature_url + "train_y_11_offline.csv", index=False)
