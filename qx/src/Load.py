import pandas as pd

def load_data_csv(fileName):
    return pd.read_csv(fileName)
def main():
    user = load_data_csv("../data/t_user.csv")
    print(user.head())

if __name__ == '__main__':
    main()