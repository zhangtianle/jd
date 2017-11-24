from sys import path
path.append('.')
from tl.src.click_feature import get_click_feature
from tl.src.loan_feature import get_loan_feature
from tl.src.order_feature import get_order_feature
from tl.src.user_feature import get_user_feature
from tl.src.util import *


root_dir, train_url, feature_url = get_url()
loan, user, order, click = read_data()

# 转换金额
loan['loan_amount'] = change_loan(loan['loan_amount'])
order['price'] = change_loan(order['price'])
order['discount'] = change_loan(order['discount'])

uid = pd.DataFrame(user["uid"])

MONTH = 10
NUM = 3.0

feature_loan = get_loan_feature(MONTH, NUM, uid, loan)
user_m = get_user_feature(MONTH, user, feature_url, save=0)
feature = get_order_feature(MONTH, NUM, order, uid)
# feature_click = get_click_feature(MONTH, click)
feature_click = pd.DataFrame(pd.read_csv('/home/kyle/project/jd/tl/feature/click_feature_10.csv'))

feature = pd.merge(feature, feature_loan, on=["uid"], how="left")
feature = pd.merge(feature, user_m, on=["uid"], how="left")
feature = pd.merge(feature, feature_click, on=["uid"], how="left")

# 处理异常值
feature = handle_na(feature)

# 保存特征数据
feature.to_csv(feature_url + "train_x_offline.csv", index=False)

# 获得预测值
loan_next_month = pd.DataFrame(pd.read_csv(root_dir + "t_loan_sum.csv"))
loan_next_month.pop("month")

loan_next_month = pd.merge(uid, loan_next_month, on=["uid"], how="left")
loan_next_month["loan_sum"] = loan_next_month["loan_sum"].fillna(0.0)
# 保存预测数据
loan_next_month.to_csv(feature_url + "train_y_offline.csv", index=False)
