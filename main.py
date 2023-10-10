from cross_validate import cross_validate
from imputise import class_specifc, all_value
from normalisation import standardise, min_max_normalise
from classifiers import k_NN, niave_bayes_gaussian, niave_bayes_multinominal
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
    best_normaliser = None

    # Combine both sets of training data
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/add_train.csv")])

    # Test KNN classifier with different magnitudes of n-fold cross validation
    # n using [5,10] and k using [10,15,20]
    # for n in [5, 10]:
    #     for k in [10, 15, 20]:
    #         for imputiser in [all_value, class_specifc]:
    #             for metric in [manhattan, euclidean]:
    #                 for normaliser in [standardise, min_max_normalise]:
    #                     print(
    #                         f"CV for {n}-fold kNN (k = {k}): \nmetric = {metric.__name__} \nimp = {imputiser.__name__} \nnormaliser = {normaliser.__name__}\n"
    #                     )
    #                     res = cross_validate(
    #                         df,
    #                         n,
    #                         imputiser,
    #                         standardise,
    #                         min_max_normalise,
    #                         k_NN,
    #                         kNN_k=k,
    #                         kNN_dist_metric=metric,
    #                     )
    #                     print(
    #                         f"Average Macro F1 for {n}-fold kNN (k = {k}):",
    #                         round(float(res), 5),
    #                         "\n",
    #                     )

    #                     if res > best_result:
    #                         best_result = res
    #                         best_classifier = k_NN
    #                         best_hyperparameters = (n, k)
    #                         best_dist_metric = metric
    #                         best_imputiser = imputiser
    #                         best_normaliser = normaliser

    # Test niave bayes classifier with different magnitudes of n-fold cross validation
    for n in [5, 10]:
        for nbtype in [niave_bayes_multinominal, niave_bayes_gaussian]:
            for imputiser in [class_specifc, all_value]:
                for normaliser in [standardise, min_max_normalise]:
                    print(f"CV for {n}-fold niave bayes, imp = {imputiser.__name__}:")
                    res = cross_validate(
                        df, n, imputiser, normaliser, gaussian_outlier_detection, nbtype
                    )
                    print(f"Average Macro F1 for {n}-fold niave bayes ({nbtype.__name__}):", round(float(res), 5), "\n")

                    if res > best_result:
                        best_result = round(res, 5)
                        best_classifier = nbtype
                        best_hyperparameters = n
                        best_normaliser = normaliser
                        best_imputiser = imputiser

    if best_classifier == k_NN:
        print("Best result:")
        print(
            f"Macro F1 = {best_result}, with {best_classifier.__name__}, using hyperparameters {best_hyperparameters}"
        )
        print(
            f"Imputiser: {best_imputiser.__name__}"
            f"Distance Metric: {best_dist_metric.__name__}"
            f"Normaliser: {best_normaliser.__name__}"
        )

    else:
        print("Best result:")
        print(
            f"Macro F1 = {best_result}, with {best_classifier.__name__}, using hyperparameters {best_hyperparameters}"
        )

    return (
        best_classifier,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
        best_normaliser
    )


def classify(
    best_classifer, best_hyperparameters, best_dist_metric, best_imputiser, best_result, best_normaliser
):
    # CLASSIFY TEST DATA USING BEST CLASSIFIER
    train_data = pd.concat(
        [pd.read_csv("data/train.csv"), pd.read_csv("data/add_train.csv")]
    )
    test_data = pd.read_csv("data/test.csv")

    if best_classifer == k_NN:
        result = k_NN(
            best_hyperparameters[1],
            best_normaliser(gaussian_outlier_detection(best_imputiser(train_data))),
            best_normaliser(test_data),
            best_dist_metric,
        )
    else:
        result = niave_bayes(
            best_normaliser(gaussian_outlier_detection(best_imputiser(train_data))),
            best_normaliser(test_data),
        )

    # Generate report based on results
    f = open("s4641154.csv", "w")
    for line in result:
        f.write(str(line))
        f.write("\n")
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
        best_normaliser
    ) = find_best_classifier()

    classify(
        best_classifer,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
        best_normaliser
    )
