import configparser
import pandas as pd
import datetime as dt
from dateutil.parser import parse
from tl.src.util import change_loan

conf = configparser.ConfigParser()
conf.read("./jd.conf")

root_dir = conf.get("local", "root_dir_local")
feature_url = conf.get("local", "feature_url")

user = pd.DataFrame(pd.read_csv(root_dir + 't_user.csv'))

# pd.to_datetime(user['active_date'])

# 计算激活日期与2016年11月1日的时间差
user['delta_time'] = user.apply(lambda x: (dt.datetime(2016, 11, 1) - parse(x['active_date'])).days,  axis=1)

# 转换金钱
user['limit'] = change_loan(user['limit'])

# 将age和sex转为onehot编码，并删除原来列
user_m = pd.get_dummies(user, columns=['age', 'sex', 'limit'])
user_m = user_m.drop('active_date', axis=1)

# 保存用户特征
user_m.to_csv(feature_url + 'user.csv')

