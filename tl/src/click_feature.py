from sys import path
path.append('.')
import pandas as pd

from tl.src.user_feature import get_user_feature
from tl.src.util import get_url, read_data, split_by_month


def get_click_feature(MONTH, click):
    click["month"] = click["click_time"].apply(split_by_month)
    click = click.loc[click["month"] <= MONTH]

    # click.drop(['month', 'click_time'], axis=1, inplace=True)
    click.drop(['click_time'], axis=1, inplace=True)

    # click['click_time'] = click.apply(lambda x: (dt.datetime(2016, MONTH + 1, 1) - parse(x['click_time'])).days, axis=1)
    click_feature = pd.get_dummies(click, columns=['pid', 'param'])
    # 各网页点击次数
    click_feature = click_feature.groupby('uid').sum().reset_index()

    click_feature.to_csv('./../feature/click_feature_{}.csv'.format(MONTH), index=False)
    return click_feature


if __name__ == "__main__":
    root_dir, train_url, feature_url = get_url()
    loan, user, order, click = read_data()
    click = get_click_feature(10, click)
    user_m = get_user_feature(10, user, feature_url, save=0)
    feature = pd.merge(user_m, click, on=["uid"], how="left")
