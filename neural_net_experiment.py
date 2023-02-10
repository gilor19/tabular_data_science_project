import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation, Reshape, Embedding
from keras.layers import Concatenate, Dropout
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
from scipy.spatial import distance_matrix
from scipy.stats import rankdata


def find_columns_to_check(df, nunique_th):
    numeric_df = df.select_dtypes(include=[np.number])
    under_th_df = numeric_df.loc[:, numeric_df.nunique() < nunique_th]
    only_int_df = under_th_df.loc[:, (under_th_df.fillna(-9999) % 1 == 0).all()]
    return only_int_df.columns


def label_encoding(catcols, train_data):
    label_dict= {}
    for col in catcols:
      le = LabelEncoder()
      train_data[col] = le.fit_transform(train_data[col])
      label_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return label_dict


def plot_loss(history):
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def preprocess(df, nunique_th, target_col_name, target_col_map):
    # find columns to be tested
    columns_to_check = find_columns_to_check(df, nunique_th)[:]

    # drop_null
    df = df.dropna()

    # convert target to 0,1
    df[target_col_name] = df[target_col_name].map(target_col_map)

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
      embedding_size = 10
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


def train_model(model, x_train, y_train, columns_to_check):
    input_train_list = []
    #input_test_list = []
    for c in columns_to_check:
        input_train_list.append(x_train[c].values)
        #input_test_list.append(x_test[c].values)
    history = model.fit(input_train_list, y_train, epochs= 10, batch_size=32, verbose=2)
    #history = model.fit(input_train_list, y_train, validation_data=(input_test_list, y_test), epochs= 10, batch_size=32, verbose=2)
    plot_loss(history)
    return


def get_embeddings(model, columns_to_check):
    embedding_dict = dict()
    for c in columns_to_check:
        embedding_dict[c] = model.get_layer(c +'_Embedding').get_weights()[0]
    return embedding_dict


def plot_embeddings_tsne(vectors, col_name):
    tsne = TSNE(n_components=2, verbose=1, perplexity=15, random_state=123, init='random', learning_rate=200)
    z = tsne.fit_transform(vectors)
    df = pd.DataFrame()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2",
                    palette=sns.color_palette("hls", 12),
                    data=df).set(title="{} Column Values Embedding T-SNE projection".format(col_name))
    plt.legend(bbox_to_anchor=(1.05, 0.95), loc='upper left', borderaxespad=0)
    plt.show()
    return


def create_original_space_dist_matrix(label_dict):
    orig_values_vec = [[k] for k in label_dict.keys()]
    euclidian_dist_matrix = distance_matrix(orig_values_vec, orig_values_vec)
    return np.apply_along_axis(convert_to_ranked_dist, 1, euclidian_dist_matrix)


def create_embedding_space_dist_matrix(embedding):
    euclidian_dist_matrix = distance_matrix(embedding, embedding)
    return np.apply_along_axis(convert_to_ranked_dist, 1, euclidian_dist_matrix)


def convert_to_ranked_dist(arr):
    return rankdata(arr).astype(int)


if __name__ == "__main__":
    # data = pd.read_csv('./datasets/adult_all.csv')
    # x_train, y_train, columns_to_check, label_dict = preprocess(data, 100, 'income', {'<=50K': 0, '>50K': 1})
    # model = build_model(x_train, columns_to_check)
    # train_model(model, x_train, y_train, columns_to_check)
    # embeddings = get_embeddings(model, columns_to_check)

    with open('embedding.pickle', 'rb') as handle:
        embeddings_loaded = pickle.load(handle)

    with open('label_dict.pickle', 'rb') as handle:
        label_dict_loaded = pickle.load(handle)

    orig_dist_matrix = create_original_space_dist_matrix(label_dict_loaded['capital-loss'])
    embedd_dist_matrix = create_embedding_space_dist_matrix(embeddings_loaded['capital-loss'])
    print("")
