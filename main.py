import pandas as pd
from neural_network_solution.neural_net_framework import preprocess, build_model, train_model, get_embeddings, \
    create_embedding_space_dist_matrix, create_original_space_dist_matrix, avg_spearmanr_for_two_dist_matrices
from datasets.edit_dataset import convert_regression_to_classification


def main(dataset_path, target_col, regression_task=False, nunique_th=50, corr_th=0.4):
    data = pd.read_csv(dataset_path)
    if regression_task:
        data = convert_regression_to_classification(data, target_col)
    x_train, y_train, columns_to_check, label_dict = preprocess(data, nunique_th, target_col)
    print(str(columns_to_check.values) + " have less than " + str(
        nunique_th) + " unique INT values, therefore they will be checked.")
    print("Training embedding model start.")
    model = build_model(x_train, columns_to_check)
    train_model(model, x_train, y_train, columns_to_check)
    embeddings = get_embeddings(model, columns_to_check)
    print("Training embedding model end.")

    for c in columns_to_check:
        orig_dist_matrix = create_original_space_dist_matrix(label_dict[c])
        embedd_dist_matrix = create_embedding_space_dist_matrix(embeddings[c])
        score = avg_spearmanr_for_two_dist_matrices(orig_dist_matrix, embedd_dist_matrix)
        print(
            "\nFor the '{}' column, Spearman correlation score of the original values space and the embedding space is {}.".format(
                c, score))
        if score > corr_th:
            print("This means '{}' column is ordinal, you should use the original values or mapping.".format(c))
        else:
            print("This means '{}' column is nominal, you should use one hot encoding.".format(c))


if __name__ == "__main__":
    dataset, target = './datasets/converted_datasets/video_games_sales_converted_regression.csv', 'Global_Sales'
    main(dataset, target, regression_task=True)
