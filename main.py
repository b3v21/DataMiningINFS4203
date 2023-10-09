from cross_validate import cross_validate
from imputise import class_specifc, all_value
from normalisation import standardise, min_max_normalise
from classifiers import k_NN, niave_bayes
from metrics import manhattan, euclidean, mc_f1
from clean import gaussian_outlier_detection
import pandas as pd

"""
This file iterates through each classifier / hyperparameter combination and
finds the best one via cross validation on the training data. Afterwards the
test data is classified using this classifier and the results are outputted.
By default this file will only run classify() as find_best_classifier() takes
a while to run on most machines. Feel free to uncomment the find_best_classifier()
in main() below.
"""


def find_best_classifier():
    # START BY FINDING BEST CLASSIFIER

    best_result = 0
    best_classifier = None
    best_hyperparameters = None
    best_imputiser = None
    best_dist_metric = None

    # Combine both sets of training data
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/add_train.csv")])

    # Test KNN classifier with different magnitudes of n-fold cross validation
    # n using [5,10] and k using [5,10,15,20]
    for n in [5, 10]:
        for k in [5, 10, 15, 20]:
            for imputiser in [class_specifc, all_value]:
                for metric in [manhattan, euclidean]:
                    print(
                        f"CV for {n}-fold kNN (k = {k}), metric = {metric.__name__}, imp = {imputiser.__name__}"
                    )
                    res = cross_validate(
                        df,
                        n,
                        imputiser,
                        standardise,
                        min_max_normalise,
                        k_NN,
                        kNN_k=k,
                        kNN_dist_metric=metric,
                    )
                    print(
                        f"Average Macro F1 for {n}-fold kNN (k = {k}):",
                        round(float(res), 3),
                        "\n",
                    )

                    if res > best_result:
                        best_result = res
                        best_classifier = k_NN
                        best_hyperparameters = (n, k)
                        best_dist_metric = metric
                        best_imputiser = imputiser

    # Test niave bayes classifier with different magnitudes of n-fold cross validation
    for n in [5, 10]:
        for imputiser in [class_specifc, all_value]:
            print(f"CV for {n}-fold niave bayes (k = {k}), imp = {imputiser.__name__}:")
            res = cross_validate(
                df, n, imputiser, standardise, min_max_normalise, niave_bayes
            )
            print(f"Average Macro F1 for {n}-fold niave bayes:", round(res, 3) + "\n")

            if res > best_result:
                best_result = round(res, 3)
                best_classifier = niave_bayes
                best_hyperparameters = n

    if best_classifier == k_NN:
        print("Best result:")
        print(
            f"\t Macro F1 = {best_result}, with {best_classifier.__name__}, using hyperparameters {best_hyperparameters}"
        )
        print(
            f"\t Imputiser: {best_imputiser.__name__}\n"
            f"Distance Metric: {best_dist_metric.__name__}"
        )

    else:
        print("Best result:")
        print(
            f"\t Macro F1 = {best_result}, with {best_classifier.__name__}, using hyperparameters {best_hyperparameters}"
        )

    return (
        best_classifier,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
    )


def classify(
    best_classifer, best_hyperparameters, best_dist_metric, best_imputiser, best_result
):
    # CLASSIFY TEST DATA USING BEST CLASSIFIER
    train_data = pd.concat(
        [pd.read_csv("data/train.csv"), pd.read_csv("data/add_train.csv")]
    )
    test_data = pd.read_csv("data/test.csv")

    if best_classifer == k_NN:
        result = k_NN(
            best_hyperparameters[1],
            min_max_normalise(gaussian_outlier_detection(best_imputiser(train_data))),
            min_max_normalise(test_data),
            min_max_normalise,
            best_dist_metric,
        )
    else:
        result = niave_bayes(
            best_hyperparameters,
            min_max_normalise(gaussian_outlier_detection(best_imputiser(train_data))),
            min_max_normalise(test_data),
        )

    # Generate report based on results
    f = open("s4641154.csv", "w")
    for key in result:
        f.write(str(result[key]) + "\n")
    f.write(str(float(best_result)))
    f.close()


if __name__ == "__main__":
    # This is turned off by default as it takes a while to run on most machines.
    # The best classifier is loaded into classify() below.
    (
        best_classifer,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
    ) = find_best_classifier()

    classify(
        best_classifer,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
    )
