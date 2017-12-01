from sys import path

path.append('../../')
import pandas as pd

from tl.src.user_feature import get_user_feature
from tl.src.util import get_url, read_data, split_by_month


def get_click_feature(start_month, MONTH, click):
    root_dir, train_url, feature_url = get_url()
    click["month"] = click["click_time"].apply(split_by_month)
    click = click[(click["month"] >= start_month) & click["month"] <= MONTH]

    # click.drop(['month', 'click_time'], axis=1, inplace=True)
    click.drop(['click_time'], axis=1, inplace=True)

    # click['click_time'] = click.apply(lambda x: (dt.datetime(2016, MONTH + 1, 1) - parse(x['click_time'])).days, axis=1)
    click_feature = pd.get_dummies(click, columns=['pid', 'param'])
    # 各网页点击次数
    click_feature = click_feature.groupby('uid').sum().reset_index()
    click_feature = click_feature.drop('month', axis=1)

    click_feature.to_csv(feature_url + 'click_feature_start_{0}_end_{1}.csv'.format(start_month, MONTH), index=False)
    return click_feature


if __name__ == "__main__":
    root_dir, train_url, feature_url = get_url()
    loan, user, order, click = read_data()
    # click = get_click_feature(10, click)
    click = get_click_feature(9, 11, click)
    # user_m = get_user_feature(10, user, feature_url, save=0)
    # feature = pd.merge(user_m, click, on=["uid"], how="left")
