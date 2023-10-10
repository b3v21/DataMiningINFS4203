import pandas as pd

from classifiers import k_NN, niave_bayes_multinominal, ensemble_kNN_nb_dt, decision_tree
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
    
    num_rows = len(df) // k
    samples = []

    shuffled_df = df.sample(frac=1)

    for i in range(k):
        if i < k - 1:
            part = shuffled_df.iloc[i * num_rows:(i + 1) * num_rows]
        else:
            part = shuffled_df.iloc[i * num_rows:]
        samples.append(part)

    mc_f1_scores = []
    for i in range(k):
        print(f"Computing fold {i + 1} of {k} (test nrows = {len(samples[i])})")
        start = t.time()
        # apply preprocessing to train data
        imputised_train_data = imp_method(pd.concat(samples[:i] + samples[i + 1 :]))
        cleaned_train_data = clean_method(imputised_train_data)
        normalised_train_data = norm_method(cleaned_train_data, cols=range(0, 100))

        # apply preprocessing to test data
        imputised_test_data = imp_method(samples[i])
        cleaned_test_data = clean_method(imputised_test_data)
        normalised_test_data = norm_method(cleaned_test_data, cols=range(0, 100))
        
        # Remove label column from test data for classifier
        test_copy_no_label = normalised_test_data.drop("Label", axis=1)

        if classifier == niave_bayes_multinominal:
            predicted_labels = classifier(normalised_train_data, test_copy_no_label)
        elif classifier == ensemble_kNN_nb_dt:
            predicted_labels = classifier(kNN_k, normalised_train_data, test_copy_no_label, kNN_dist_metric)
        elif classifier == k_NN:
            predicted_labels = classifier(
                kNN_k,
                normalised_train_data,
                test_copy_no_label,
                kNN_dist_metric,
            )
        elif classifier == decision_tree:
            predicted_labels = classifier(
                normalised_train_data,
                test_copy_no_label,
            )
        true_labels = normalised_test_data["Label"]

        mc_f1_res = mc_f1(true_labels, predicted_labels)
        mc_f1_scores.append(mc_f1_res)
        end = t.time()
        print(f"Macro F1: {round(mc_f1_res,5)}")
        print(f"Elapsed: {round(end - start,1)} seconds")

    return sum(mc_f1_scores) / k
