from sys import path
path.append('../../')
import pandas as pd
import datetime as dt
from dateutil.parser import parse
from tl.src.util import change_loan, get_url, read_data


def get_user_feature(start_month, MONTH, user, feature_url, save=0):
    # 计算激活日期与2016年x月1日的时间差
    user['delta_time'] = user.apply(lambda x: (dt.datetime(2016, MONTH + 1, 1) - parse(x['active_date'])).days, axis=1)
    user['start_delta_time'] = user.apply(lambda x: (dt.datetime(2016, start_month, 1) - parse(x['active_date'])).days, axis=1)

    # 转换金钱
    user['limit'] = change_loan(user['limit'])

    # 将age和sex转为onehot编码，并删除原来列
    user_m = pd.get_dummies(user, columns=['age', 'sex', 'limit'])
    user_m = user_m.drop('active_date', axis=1)

    # 保存用户特征
    if save == 1:
        user_m.to_csv(feature_url + 'user.csv')

    return user_m


if __name__ == '__main__':
    root_dir, train_url, feature_url = get_url()
    loan, user, order, click = read_data()
    get_user_feature(10, user, feature_url, 1)
