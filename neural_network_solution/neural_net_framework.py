import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape, Embedding, Concatenate, Dropout
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr


def convert_regression_to_classification(data, target_col, save_path=False):
    data[target_col] = data[target_col].apply(lambda x: 1 if x > data[target_col].median() else 0)
    if save_path:
        data.to_csv(save_path, index=False)
    return data


def find_columns_to_check(df, target_col_name, nunique_th):
    numeric_df = df.select_dtypes(include=[np.number])
    under_th_df = numeric_df.loc[:, numeric_df.nunique() <= nunique_th]
    only_int_df = under_th_df.loc[:, (under_th_df.fillna(-9999) % 1 == 0).all()]
    more_than_two_values = only_int_df.loc[:, only_int_df.nunique() > 2]
    if target_col_name in more_than_two_values:
        return more_than_two_values.columns.drop(target_col_name)
    else:
        return more_than_two_values.columns


def label_encoding(catcols, train_data):
    label_dict= {}
    for col in catcols:
      le = LabelEncoder()
      train_data[col] = le.fit_transform(train_data[col])
      label_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return label_dict


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def preprocess(df, nunique_th, target_col_name):
    # remove columns with too many null values
    columns_to_keep = df.columns[df.isna().mean() < 0.8]
    df = df[columns_to_keep]


    # find columns to be tested
    columns_to_check = find_columns_to_check(df, target_col_name, nunique_th)[:]

    # for col in columns_to_check:
    #     if check_high_percentage_of_category(df, col, threshold=0.7):
    #         columns_to_check = columns_to_check.drop(col)
    #
    #         print("{} column was dropped due to values imbalance".format(col))

    # drop rows with any null value
    df = df.dropna(subset=columns_to_check)


    # split to x,y
    y_train = df[target_col_name]
    x_train = df.drop([target_col_name], axis=1)

    # map tested columns to labels
    label_dict = label_encoding(columns_to_check, x_train)

    return x_train, y_train, columns_to_check, label_dict


def build_model(x_train, columns_to_check):
    input_models=[]
    output_embeddings=[]
    for c in columns_to_check:
      cat_emb_name= c +'_Embedding'
      no_of_unique_cat  = x_train[c].nunique()
      embedding_size = min(int(no_of_unique_cat/2),50)
      input_model = Input(shape=(1,),name = c+'_Input')
      output_model = Embedding(no_of_unique_cat, embedding_size,name=cat_emb_name)(input_model)
      output_model = Reshape(target_shape=(embedding_size,))(output_model)
      input_models.append(input_model)
      output_embeddings.append(output_model)

    output = Concatenate()(output_embeddings)
    output = Dense(512, kernel_initializer="uniform")(output)
    output = Activation('relu')(output)
    output= Dropout(0.4)(output)
    output = Dense(256, kernel_initializer="uniform")(output)
    output = Activation('relu')(output)
    output= Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=input_models, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, columns_to_check, plot_loss=False):
    input_train_list = []
    for c in columns_to_check:
        input_train_list.append(x_train[c].values)
    history = model.fit(input_train_list, y_train, epochs=10, batch_size=32, verbose=2)
    if plot_loss:
        plot_loss(history)
    return


def get_embeddings(model, columns_to_check):
    embedding_dict = dict()
    for c in columns_to_check:
        embedding_dict[c] = model.get_layer(c +'_Embedding').get_weights()[0]
    return embedding_dict


def create_original_space_dist_matrix(label_dict):
    orig_values_vec = [[k] for k in label_dict.keys()]
    return distance_matrix(orig_values_vec, orig_values_vec)


def create_embedding_space_dist_matrix(embedding):
    return distance_matrix(embedding, embedding)


def avg_spearmanr_for_two_dist_matrices(original_space_dist_matrix, embedding_space_dist_matrices):
    spermanr_sums = 0
    n = original_space_dist_matrix.shape[0]
    for row in range(n):
        orig_space_row = original_space_dist_matrix[row]
        embedd_space_row = embedding_space_dist_matrices[row]
        spermanr_sums += spearmanr(orig_space_row, embedd_space_row).correlation
    return spermanr_sums / n


def load_existing_embeddings_for_dev(emb_path, label_dict_path):
    with open(emb_path, 'rb') as handle:
        embeddings = pickle.load(handle)

    with open(label_dict_path, 'rb') as handle:
        label_dict = pickle.load(handle)

    return embeddings, label_dict


def main(dataset_path, target_col, nunique_th, corr_th):
    data = pd.read_csv(dataset_path)
    x_train, y_train, columns_to_check, label_dict = preprocess(data, nunique_th, target_col)
    print(str(columns_to_check.values) + " have less than " + str(nunique_th) + " unique INT values, therefore they will be checked.")
    print("Training embedding model start.")
    model = build_model(x_train, columns_to_check)
    train_model(model, x_train, y_train, columns_to_check)
    embeddings = get_embeddings(model, columns_to_check)
    print("Training embedding model end.")

    for c in columns_to_check:
        orig_dist_matrix = create_original_space_dist_matrix(label_dict[c])
        embedd_dist_matrix = create_embedding_space_dist_matrix(embeddings[c])
        score = avg_spearmanr_for_two_dist_matrices(orig_dist_matrix, embedd_dist_matrix)
        print("\nFor the '{}' column, Spearman correlation score of the original values space and the embedding space is {}.".format(c,score))
        if score > corr_th:
            print("This means '{}' column is ordinal, you should use the original values or mapping.".format(c))
        else:
            print("This means '{}' column is nominal, you should use one hot encoding.".format(c))


if __name__ == "__main__":
    main('../datasets/converted_datasets/video_games_sales_converted_classification.csv', 'Global_Sales', 50, 0.4)
    print("")
