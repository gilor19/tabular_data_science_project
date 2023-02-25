import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
from statsmodels.distributions.empirical_distribution import ECDF


def find_numeric_columns_to_check(df, nunique_th):
    numeric_df = df.select_dtypes(include=[np.number])
    under_th_df = numeric_df.loc[:, numeric_df.nunique() < nunique_th]
    only_int_df = under_th_df.loc[:, (under_th_df.fillna(-9999) % 1 == 0).all()]
    more_than_two_values = only_int_df.loc[:, only_int_df.nunique() > 2]
    return list(more_than_two_values.columns)


def get_dummies(columns):
    dummies = {}
    for column in columns:
        dummy_variable = pd.get_dummies(data[column])
        prefix = f"{column}_"
        dummy_variable = dummy_variable.add_prefix(prefix)
        dummies[column] = dummy_variable
        data[dummy_variable.columns] = dummy_variable
    return dummies


# def plot(check_columns):
#     coefs = []
#
#     for idx, df in enumerate(dummies):
#         dummy_coefs = model.coef_[-df.shape[1]:]
#         df = df.rename(columns=lambda x: re.sub(r'[^0-9]+', '', x))
#         dummy_columns = df.columns
#         coef_df = pd.DataFrame({'dummy_variable': dummy_columns, 'coefficient': dummy_coefs.flatten()})
#         corr, _ = spearmanr(coef_df['dummy_variable'], coef_df['coefficient'])
#         reg = LinearRegression()
#         x = coef_df['dummy_variable'].to_numpy().reshape(-1, 1)
#         reg.fit(x, coef_df['coefficient'])
#         r2 = r2_score(coef_df['coefficient'],  reg.predict(x))
#         coefs.append(coef_df)
#         print(coef_df)
#         axis[idx % width, int(idx / width)].plot(coef_df['dummy_variable'], coef_df['coefficient'], markersize=12,
#                                                  linewidth=1, linestyle='dashed', label=f"R^2: {round(r2, 2)}")
#         axis[idx % width, int(idx / width)].plot(coef_df['dummy_variable'], reg.predict(x),
#                                                  color='red', label=f'{check_columns[idx]}',)
#         axis[idx % width, int(idx / width)].legend(loc='best')


if __name__ == "__main__":
    data = pd.read_csv('../datasets/converted_datasets/titanic_converted.csv')
    # target = 'income'  # adults
    # target = 'Global_Sales'  # video game sales
    # target = 'mode'   # spotify_converted
    target = 'Survived'  # titanic_converted
    print("data shape: ", data.shape)
    data['randNumCol'] = np.random.choice([2, 1, 0.5, 0.2], data.shape[0])  # test if our filter worked
    columns_to_check = find_numeric_columns_to_check(data, 50)
    assert 'randNumCol' not in columns_to_check
    data = data.select_dtypes(exclude=['object'])
    print("data shape without dtypes obj: ", data.shape)
    data = data.dropna()
    print("data shape without na: ", data.shape)
    y = data[target]
    dummies = get_dummies(columns_to_check)

    for col_name in dummies:
        X = dummies[col_name]
        original_column = data[col_name]
        model = LinearRegression().fit(X, y)
        coefs = model.coef_
        order = sorted(original_column.unique().reshape(-1, 1))
        B = LinearRegression().fit(order, coefs).coef_
        hist = []
        for i in range(1000):
            random.shuffle(order)
            beta_i = LinearRegression().fit(order, coefs).coef_
            hist.append(beta_i)
        hist = [float(arr) for arr in hist]
        ecdf = ECDF(hist)
        prob = 1 - ecdf(B)
        print(f"The probability of sampling a value greater than or equal to {float(B)} is {float(prob)}")
        if prob < 0.05:
            print(f"{col_name} is Ordinal")
        else:
            print(f"{col_name} is Nominal")

    # data = data.drop(columns_to_check, axis=1)
    # data = data.drop(target, axis=1)
    # print("data shape with dummies: ", data.shape)
    # X = data
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # model = LinearRegression()
    # model.fit(X, y)
    #
    # # Access the coefficients for each dummy variable
    #
    # height, width = 5, 4
    # figure, axis = plt.subplots(width, height, figsize=(10, 5))
    #
    # plot(columns_to_check)

    # Combine all the operations and display
    # plt.show()
    # print()
    # est = sm.OLS(y, X)
    # est2 = est.fit()
    # print(est2.summary())

    # print()
# שאלות:
#
# 1 - איך להתמודד עם בעיות שהY הוא קלאסיפיקציה ולא רגרסיה?
# 2 - באיזה מטריקה להשתמש כדי להסיק על הB?