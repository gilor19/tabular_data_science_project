import pandas as pd
import numpy as np


def find_columns_to_check(df, nunique_th):
    numeric_df = df.select_dtypes(include=[np.number])
    under_th_df = numeric_df.loc[:, numeric_df.nunique() < nunique_th]
    only_int_df = under_th_df.loc[:, (under_th_df.fillna(-9999) % 1 == 0).all()]
    return only_int_df.columns


if __name__ == "__main__":
    data = pd.read_csv('./datasets/house_price_train.csv')
    data['randNumCol'] = np.random.choice([2,1,0.5,0.2], data.shape[0])
    columns_to_check = find_columns_to_check(data, 50)
    print("")
