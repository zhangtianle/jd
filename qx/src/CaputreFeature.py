import Load
from dateutil.parser import parse
import pandas as pd


def split_by_month(data):
    return parse(data).month

def count_price_per_order(column):
    return column["price"]*column["qty"]-column["discount"]
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
def main():
    # 提取用户信息
    user = Load.load_data_csv("../data/t_user.csv")
    uid = pd.DataFrame(user["uid"])
    # 用户特征
    user["sex"] = user["sex"].apply(lambda x: 1 if x == 1 else 0)
    user.pop("active_date")
    MONTHES = [8,9,10]
    for MONTH in MONTHES:
        #提取历史贷款信息
        loan = Load.load_data_csv("../data/t_loan.csv")
        loan["month"] = loan["loan_time"].apply(split_by_month)
        #特征
        loan["pay_per_month"] = loan.apply(get_pay_per_month, axis=1)
        loan["remain_loan"] = loan.loc[loan["month"]<=MONTH].apply(get_remain_loan, axis=1, args=(MONTH,))
        loan["remain_loan"] = loan["remain_loan"] .fillna(0)

        loan["remain_pay"] = loan.loc[loan["month"] <= MONTH].apply(get_remain_pay, axis=1, args=(MONTH,))
        loan["remain_pay"] = loan["remain_pay"].fillna(0)
        #当月贷款
        loan_per_month =  pd.DataFrame( loan.loc[loan["month"]==MONTH]["loan_amount"].groupby([loan["uid"]]).sum()).reset_index()
        #当月月供
        pay_per_month = pd.DataFrame(loan.loc[loan["month"]==MONTH]["pay_per_month"].groupby([loan["uid"]]).sum()).reset_index()
        #历史贷款总额
        remain_loan = pd.DataFrame(loan.loc[loan["month"]<=MONTH]["remain_loan"].groupby([loan["uid"]]).sum()).reset_index()
        #累计月供
        remain_pay = pd.DataFrame(loan.loc[loan["month"]<=MONTH]["remain_pay"].groupby([loan["uid"]]).sum()).reset_index()

        feature_loan = pd.merge(uid, loan_per_month, on=["uid"], how="left")
        feature_loan = pd.merge(feature_loan, pay_per_month, on=["uid"], how="left")
        feature_loan = pd.merge(feature_loan, remain_loan, on=["uid"], how="left")
        feature_loan = pd.merge(feature_loan, remain_pay, on=["uid"], how="left")

        #提取购物特征
        order = Load.load_data_csv("../data/t_order.csv")
        order["price"] = order["price"].fillna(0)
        #为消费记录，按照时间分割
        order["month"] = order["buy_time"].apply(split_by_month)
        #获取用户在每笔费用的实际消费（金钱*数量-折扣）
        order["real_price"] = order.apply(count_price_per_order, axis=1)
        #获取每个用户当月的总消费额
        feature_order = pd.DataFrame({"total_price":order.loc[order["month"]==MONTH]["real_price"].groupby([order["uid"]]).sum()}).reset_index()


        #合并数据集，将User和特种一一对应
        feature = pd.merge(uid,user,on=["uid"],how="left")
        feature = pd.merge(feature, feature_order, on=["uid"], how="left")
        feature = pd.merge(feature,feature_loan,on=["uid"],how="left")

        #处理异常值
        feature["month"] = MONTH
        feature["total_price"] = feature["total_price"].fillna(0.0)
        feature["loan_amount"] = feature["loan_amount"].fillna(0.0)
        feature["pay_per_month"] = feature["pay_per_month"].fillna(0.0)
        feature["remain_loan"] = feature["remain_loan"].fillna(0.0)
        feature["remain_pay"] = feature["remain_pay"].fillna(0.0)
        #保存特征数据
        feature.to_csv("../train/feature_{}.csv".format(MONTH),index=False)
        #获得预测值
        loan_next_month = pd.DataFrame(
            loan.loc[loan["month"] == MONTH+1]["loan_amount"].groupby([loan["uid"]]).sum()).reset_index()
        loan_next_month = pd.merge(uid, loan_next_month, on=["uid"], how="left")
        loan_next_month["loan_amount"] = loan_next_month["loan_amount"].fillna(0.0)
        #保存预测数据
        loan_next_month.to_csv("../train/loan_next_month_{}.csv".format(MONTH),index=False)
if __name__ == '__main__':
    main()