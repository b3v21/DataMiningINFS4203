import pandas as pd

from imputise import class_specifc, all_value
from normalisation import standardise, min_max_normalise
from clean import gaussian_outlier_detection
from classifiers import k_NN, niave_bayes
from metrics import manhattan, mc_f1
import time as t


def cross_validate(
    df,
    k,
    imp_method,
    norm_method,
    clean_method,
    classifier,
    kNN_k=15,
    kNN_dist_metric=manhattan,
):
    partition_percentage = 1 / k
    samples = []

    for _ in range(k):
        partition = df.sample(frac=partition_percentage, ignore_index=False)
        df = df.drop(partition.index)
        samples.append(partition)

    mc_f1_scores = []
    for i in range(k):
        print(f"Computing fold {i + 1} of {k}")
        start = t.time()
        # apply preprocessing to train data
        imputised_train_data = imp_method(pd.concat(samples[:i] + samples[i + 1 :]))
        cleaned_train_data = clean_method(imputised_train_data)
        normalised_train_data = norm_method(cleaned_train_data, cols=range(0, 100))

        # apply preprocessing to test data
        imputised_test_data = imp_method(samples[i])
        cleaned_test_data = clean_method(imputised_test_data)
        normalised_test_data = norm_method(cleaned_test_data, cols=range(0, 100))

        if classifier == niave_bayes:
            predicted_labels = classifier(normalised_train_data, normalised_test_data)
        else:
            predicted_labels = classifier(
                kNN_k,
                normalised_train_data,
                normalised_test_data,
                norm_method,
                kNN_dist_metric,
            )
        true_labels = normalised_test_data["Label"].to_dict()

        mc_f1_res = mc_f1(list(true_labels.values()), list(predicted_labels.values()))
        mc_f1_scores.append(mc_f1_res)
        end = t.time()
        print(f"Macro F1: {round(mc_f1_res,3)}")
        print(f"Elapsed: {round(end - start,0)} seconds")

    return sum(mc_f1_scores) / k
