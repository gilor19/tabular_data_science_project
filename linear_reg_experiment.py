import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def find_numeric_columns_to_check(df, nunique_th):
    numeric_df = df.select_dtypes(include=[np.number])
    under_th_df = numeric_df.loc[:, numeric_df.nunique() < nunique_th]
    only_int_df = under_th_df.loc[:, (under_th_df.fillna(-9999) % 1 == 0).all()]
    return list(only_int_df.columns)


if __name__ == "__main__":
    data = pd.read_csv('./datasets/house_price_train.csv')
    data['randNumCol'] = np.random.choice([2, 1, 0.5, 0.2], data.shape[0])  # test if our filter worked
    columns_to_check = find_numeric_columns_to_check(data, 50)
    assert 'randNumCol' not in columns_to_check
    check_column = 'OverallCond'
    target = 'SalePrice'
    dummy_variable = pd.get_dummies(data[check_column])

    # Concatenate the original data and the dummy variables
    data = pd.concat([data, dummy_variable], axis=1)
    # Define the features and target variable
    y = data[target]
    X = data[columns_to_check]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Access the coefficients for each dummy variable
    dummy_coefs = model.coef_[-dummy_variable.shape[1]:]
    dummy_columns = dummy_variable.columns
    coef_df = pd.DataFrame({'dummy_variable': dummy_columns, 'coefficient': dummy_coefs})
    ax1 = coef_df.plot.scatter(x='dummy_variable', y='coefficient', c='DarkBlue')
    print(coef_df)
