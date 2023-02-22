import pandas as pd

# datasets edit details:

# adult
# columns_to_convert_randomly: ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']
# columns_to_convert_orderly = ['education']
# order_map = {'education': {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7,
#                  'HS-grad': 8, 'Prof-school': 9,'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college': 12, 'Bachelors': 13
#                 , 'Masters': 14, 'Doctorate': 15}}
# target = 'income'
# target_map = {'<=50K': 0, '>50K': 1}

# spotify
# columns_to_convert_orderly = []
# order_map = {}
# columns_to_convert_randomly = ['genre']

# video games
# columns_to_convert_orderly = []
# order_map = {}
# columns_to_convert_randomly = ['Platform', 'Genre']

# titanic
# columns_to_convert_orderly=[]
# order_map = {}
# columns_to_convert_randomly = ['Embarked']


def convert_str_cat_columns_to_numeric(df, cols_to_convert_randomly, cols_to_convert_orderly, order_map_dict, save_path,
                                       target_col=None, target_col_map=None):
    for col in cols_to_convert_orderly:
        df[col] = df[col].map(order_map_dict[col])

    for col in cols_to_convert_randomly:
        df[col] = pd.Categorical(df[col])
        df['code'] = df[col].cat.codes
        df = df.drop([col], axis=1)
        df[col] = df['code']
        df = df.drop(['code'], axis=1)

    if target_col:
        df[target_col] = df[target_col].map(target_col_map)

    df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    # example run
    data = pd.read_csv('./original_datasets/adult.csv')
    columns_to_convert_randomly: ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']
    columns_to_convert_orderly = ['education']
    order_map = {'education': {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6,
                               '12th': 7, 'HS-grad': 8, 'Prof-school': 9,'Assoc-acdm': 10, 'Assoc-voc': 11,
                               'Some-college': 12, 'Bachelors': 13, 'Masters': 14, 'Doctorate': 15}}
    target = 'income'
    target_map = {'<=50K': 0, '>50K': 1}
    save_path = './converted_datasets/adults_converted_example.csx'
    convert_str_cat_columns_to_numeric(data, columns_to_convert_randomly, columns_to_convert_orderly, order_map,
                                       save_path, target, target_map)


