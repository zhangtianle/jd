from sys import path
path.append('../../')
from tl.src.util import count_price_per_order
import pandas as pd
from math import log


def get_order_loan(order, loan, start_month, MONTH, NUM):

    # TODO
    order["real_price"] = order.apply(count_price_per_order, axis=1)
    current_price_sum = pd.DataFrame({"current_price_sum": order.loc[order["month"] == MONTH]["real_price"].groupby(
        [order["uid"]]).sum()}).reset_index()
    current_price_sum["current_price_sum"] = current_price_sum["current_price_sum"].apply(lambda x: log(x + 1, 5))

    current_loan_sum = pd.DataFrame({"current_loan_sum": loan.loc[loan["month"] == MONTH]["loan_amount"].groupby(
        [loan["uid"]]).sum()}).reset_index()
    current_loan_sum["current_loan_sum"] = current_loan_sum["current_loan_sum"].apply(lambda x: log(x + 1, 5))

    loan_order = pd.merge(current_price_sum, current_loan_sum, on="uid", how="left")
    loan_order = loan_order.fillna(0)
    loan_order["loan_order_ratio"] = loan_order.apply(lambda x: x["current_loan_sum"] / (x["current_price_sum"] + 1),
                                                      axis=1)

    loan_order = pd.DataFrame(loan_order[["uid", "loan_order_ratio"]])
    return loan_order
