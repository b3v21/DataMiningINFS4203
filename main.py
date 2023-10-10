from cross_validate import cross_validate
from imputise import class_specific, all_value
from normalisation import standardise, min_max_normalise
from classifiers import k_NN, niave_bayes_multinominal, ensemble_kNN_nb_dt, decision_tree
from metrics import manhattan, euclidean
from clean import gaussian_outlier_detection, isolation_forest
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
    best_cleaner = None
    best_acc = 0

    # Combine both sets of training data
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/add_train.csv")])

    # Test niave bayes classifier with different magnitudes of n-fold cross validation
    for n in [5, 10]:
        for normaliser in [standardise, min_max_normalise]:
            for cleaner in [gaussian_outlier_detection, isolation_forest]:
                print(
                        f"CV for {n}-fold niave bayes: \nimp = {class_specific.__name__} \nnormaliser = {normaliser.__name__} \ncleaner = {cleaner.__name__}\n"
                    )
                f1, acc = cross_validate(
                    df, n, class_specific, normaliser, cleaner, niave_bayes_multinominal
                )
                print(f"Average Macro F1:", round(float(f1), 5))
                print(f"Average Accuracy:", round(float(acc), 5), "\n")

                if f1 > best_result:
                    best_result = round(f1, 5)
                    best_acc = round(acc,5)
                    best_classifier = niave_bayes_multinominal
                    best_hyperparameters = n
                    best_normaliser = normaliser
                    best_imputiser = class_specific
                    best_cleaner = cleaner
    
    # Test decision tree classifier with different magnitudes of n-fold cross validation
    for n in [5, 10]:
        for normaliser in [standardise, min_max_normalise]:
            for cleaner in [gaussian_outlier_detection, isolation_forest]:
                print(
                        f"CV for {n}-fold decision-tree: \nimp = {class_specific.__name__} \nnormaliser = {normaliser.__name__} \ncleaner = {cleaner.__name__}\n"
                    )
                f1, acc = cross_validate(
                    df, n, class_specific, normaliser, cleaner, decision_tree
                )
                print(f"Average Macro F1:", round(float(f1), 5))
                print(f"Average Accuracy:", round(float(acc), 5), "\n")

                if f1 > best_result:
                    best_result = round(f1, 5)
                    best_classifier = decision_tree
                    best_hyperparameters = n
                    best_normaliser = normaliser
                    best_imputiser = class_specific
                    best_cleaner = cleaner
                    best_acc = round(acc,5)
    
    # Test KNN classifier with different magnitudes of n-fold cross validation and different k values
    for n in [5, 10]:
        for k in [10, 20]:
            for metric in [manhattan, euclidean]:
                for normaliser in [standardise, min_max_normalise]:
                    for cleaner in [gaussian_outlier_detection, isolation_forest]:
                        print(
                            f"CV for {n}-fold kNN (k = {k}): \nmetric = {metric.__name__} \nimp = {class_specific.__name__} \nnormaliser = {normaliser.__name__} \ncleaner = {cleaner.__name__}\n"
                        )
                        f1, acc = cross_validate(
                            df,
                            n,
                            class_specific,
                            normaliser,
                            cleaner,
                            k_NN,
                            kNN_k=k,
                            kNN_dist_metric=metric,
                        )
                        print(
                            f"Average Macro F1:",
                            round(float(f1), 5),
                        )
                        print(f"Average Accuracy:", round(float(acc), 5), "\n")
                        if f1 > best_result:
                            best_result = round(f1, 5)
                            best_classifier = k_NN
                            best_hyperparameters = (n, k)
                            best_dist_metric = metric
                            best_imputiser = class_specific
                            best_normaliser = normaliser
                            best_cleaner = cleaner
                            best_acc = round(acc,5)
                        
    # Test ensemble classifer with different magnitudes of n-fold cross validation and different k values
    for n in [5,10]:
        for k in [10,20,40,60]:
            for normaliser in [min_max_normalise, standardise]:
                for metric in [manhattan, euclidean]:
                    for cleaner in [isolation_forest, gaussian_outlier_detection]:
                        print(f"CV for {n}-fold kNN-NB-dt Ensemble (k = {k}), \nimp = {class_specific.__name__} \nmetric = {metric.__name__} \nnormaliser = {normaliser.__name__} \ncleaner = {cleaner.__name__}\n")
                        f1, acc = cross_validate(
                            df, n, class_specific, normaliser, cleaner, ensemble_kNN_nb_dt, kNN_k=k, kNN_dist_metric=metric
                        )
                        print(f"Average Macro F1:", round(float(f1), 5))
                        print(f"Average Accuracy:", round(float(acc), 5), "\n")
                        
                        if f1 > best_result:
                            best_result = round(f1, 5)
                            best_classifier = ensemble_kNN_nb_dt
                            best_hyperparameters = (n,k)
                            best_normaliser = normaliser
                            best_imputiser = class_specific
                            best_dist_metric = metric
                            best_cleaner = cleaner
                            best_acc = round(acc,5)
            

    if best_classifier == k_NN or best_classifier == ensemble_kNN_nb_dt:
        print("Best result:")
        print(
            f"Macro F1 = {best_result}, Accuracy = {best_acc}, using hyperparameters n = {best_hyperparameters[0]}, k = {best_hyperparameters[1]}"
        )
        print(
            f"Classifier: {best_classifier.__name__} \n"
            f"Imputiser: {best_imputiser.__name__} \n"
            f"Distance Metric: {best_dist_metric.__name__}\n"
            f"Normaliser: {best_normaliser.__name__}\n"
            f"Cleaner: {best_cleaner.__name__}\n"
        )

    else:
        print("Best result:")
        print(
            f"Macro F1 = {best_result}, Accuracy = {best_acc}, using hyperparameters n = {best_hyperparameters}"
        )
        print(
            f"Classifier: {best_classifier.__name__}\n"
            f"Imputiser: {best_imputiser.__name__}\n"
            f"Normaliser: {best_normaliser.__name__}\n"
            f"Cleaner: {best_cleaner.__name__}\n"
        )

    return (
        best_classifier,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
        best_normaliser,
        best_cleaner,
        best_acc
    )


def classify(
    best_classifer, best_hyperparameters, best_dist_metric, best_imputiser, best_result, best_normaliser, best_cleaner, best_acc
):
    # CLASSIFY TEST DATA USING BEST CLASSIFIER
    train_data = pd.concat(
        [pd.read_csv("data/train.csv"), pd.read_csv("data/add_train.csv")]
    )
    test_data = pd.read_csv("data/test.csv")

    if best_classifer == k_NN:
        result = k_NN(
            best_hyperparameters[1],
            best_normaliser(best_cleaner(best_imputiser(train_data)), cols=range(0, 100)),
            best_normaliser(test_data, cols=range(0, 100)),
            best_dist_metric,
        )
    elif best_classifer == niave_bayes_multinominal:
        result = best_classifer(
            best_normaliser(best_cleaner(best_imputiser(train_data)), cols=range(0, 100)),
            best_normaliser(test_data, cols=range(0, 100)),
        )
    elif best_classifer == ensemble_kNN_nb_dt:
        result = best_classifer(
            best_hyperparameters[1],
            best_normaliser(best_cleaner(best_imputiser(train_data)), cols=range(0, 100)),
            best_normaliser(test_data, cols=range(0, 100)),
            best_dist_metric
        )
    elif best_classifer == decision_tree:
        result = best_classifer(
            best_normaliser(best_cleaner(best_imputiser(train_data)), cols=range(0, 100)),
            best_normaliser(test_data, cols=range(0, 100)),
        )

    # Generate report based on results
    df = pd.DataFrame([str(line) for line in result])
    df2 = pd.DataFrame([[str(round(float(best_acc),3)), str(round(float(best_result),3))]])
    
    df3 = pd.concat([df, df2]) 
    df3.to_csv("s4641154.csv", index=False, header=False)

    print(f"Output file generated:")
    print(
            f"Classifier: {best_classifier.__name__}\n"
            f"Imputiser: {best_imputiser.__name__}\n"
            f"Normaliser: {best_normaliser.__name__}\n"
            f"Cleaner: {best_cleaner.__name__}\n"
            f"Distance Metric: {best_dist_metric.__name__}\n\n"
        )
    print(f"Macro F1 = {round(float(best_result),3)}, Accuracy = {round(float(best_acc),3)}")

if __name__ == "__main__":
    # This is turned off by default as it takes a while to run on most machines.
    # The best classifier is loaded into classify() below.
    (
        best_classifer,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
        best_normaliser,
        best_cleaner,
        best_acc
    ) = find_best_classifier()
    
    # best_classifer = ensemble_kNN_nb_dt
    # best_hyperparameters = (5,10)
    # best_dist_metric = manhattan
    # best_imputiser = class_specific
    # best_result = 0.968
    # best_normaliser = standardise
    # best_cleaner = isolation_forest

    classify(
        best_classifer,
        best_hyperparameters,
        best_dist_metric,
        best_imputiser,
        best_result,
        best_normaliser,
        best_cleaner,
        best_acc
    )
