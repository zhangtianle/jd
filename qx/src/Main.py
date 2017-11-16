import Load
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
def split_by_month(data):
    return data.month


def main():
    MONTHS = [8,9,10]
    X = pd.DataFrame()
    Y = pd.DataFrame()
    for month in MONTHS:
        x = Load.load_data_csv("../train/feature_{}.csv".format(month))
        y = Load.load_data_csv("../train/loan_next_month_{}.csv".format(month))

        X = pd.concat([X,x])
        Y = pd.concat([Y,y])
    Test = Load.load_data_csv("../test/feature_{}.csv".format(11))
    Test.pop("uid")
    Test.pop("month")
    X.pop("uid")
    X.pop("month")
    Y.pop("uid")

    clf = LinearRegression()
    clf.fit(X,Y)

    predict = clf.predict(Test)
    for i in range(len(predict)):
        if predict[i][0] < 0:
            predict[i][0] = 0
    sample = pd.read_csv("../result_sample/Loan_Forecasting_Upload_Sample.csv", header=None)

    sample[1] = predict

    sample.to_csv("../result/Loan_Forecasting_Upload.csv",header=None,index=False,encoding = "utf-8",sep=",")




if __name__ == "__main__":
    main()
