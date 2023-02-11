import pandas as pd


def convert_str_cat_columns_to_numeric(df, cols_to_convert, save_path, education_map):
    df['education'] = df['education'].map(education_map)
    for col in cols_to_convert:
        df[col] = pd.Categorical(df[col])
        df['code'] = df[col].cat.codes
        df = df.drop([col], axis=1)
        df[col] = df['code']
        df = df.drop(['code'], axis=1)
    df.to_csv(save_path)
    return df


if __name__ == "__main__":
    cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']
    education_map = {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7,
                     'HS-grad': 8, 'Prof-school': 9,'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college': 12, 'Bachelors': 13
                    , 'Masters': 14, 'Doctorate': 15}
    data = pd.read_csv('./datasets/adult.csv')
    convert_str_cat_columns_to_numeric(data, cols, './datasets/adults_converted.csv', education_map)
