import pandas as pd
from neural_network_solution.neural_net_framework import preprocess, build_model, train_model, get_embeddings, \
    create_embedding_space_dist_matrix, create_original_space_dist_matrix, avg_spearmanr_for_two_dist_matrices, \
    convert_regression_to_classification


def main(dataset_path, target_col, regression_task=False, nunique_th=50, corr_th=0.4):
    data = pd.read_csv(dataset_path)
    if regression_task:
        data = convert_regression_to_classification(data, target_col)
    x_train, y_train, columns_to_check, label_dict = preprocess(data, nunique_th, target_col)
    print("The following columns have less than " + str(
        nunique_th) + " unique INT values, therefore they will be checked:")
    for c in columns_to_check:
        print(c)
    print("")
    print("Training embedding model start.\n\n")
    model = build_model(x_train, columns_to_check)
    train_model(model, x_train, y_train, columns_to_check)
    embeddings = get_embeddings(model, columns_to_check)
    print("\nTraining embedding model end.\n")

    for c in columns_to_check:
        orig_dist_matrix = create_original_space_dist_matrix(label_dict[c])
        embedd_dist_matrix = create_embedding_space_dist_matrix(embeddings[c])
        score = round(avg_spearmanr_for_two_dist_matrices(orig_dist_matrix, embedd_dist_matrix),2)
        print(
            "\n'{}': Spearman correlation score of the original values space and the embedding space is {}.".format(
                c, score))
        if score > corr_th:
            print("This means '{}' column is ordinal, you should use the original values as mapping.\n".format(c))
        else:
            print("This means '{}' column is nominal, you should use one hot encoding.\n".format(c))


if __name__ == "__main__":
    dataset, target = './datasets/converted_datasets/video_games_sales_converted.csv', 'Global_Sales'
    main(dataset, target, regression_task=True)
