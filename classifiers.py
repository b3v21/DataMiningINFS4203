import pandas as pd
from normalisation import standardise
from metrics import manhattan
import math
import numpy as np


def one_hot_encode(data):
    for col in data.columns[100:]:
        if col == "Label":
            continue
        data = pd.concat(
            [data, pd.get_dummies(data[col], prefix=col).astype(int)], axis=1
        )
        data = data.drop(col, axis=1)
    return data


def k_NN(k, train_data, test_data, normalisation_method, distance_metric):
    train_data = one_hot_encode(train_data)
    test_data = one_hot_encode(test_data)

    for col in test_data.columns:
        if col not in train_data.columns:
            train_data[col] = 0

    for col in train_data.columns:
        if col not in test_data.columns:
            test_data[col] = 0

    train_data = normalisation_method(train_data, cols=range(100, 128))
    test_data = normalisation_method(test_data, cols=range(100, 128))

    train_data = train_data.reindex(sorted(train_data.columns), axis=1)
    test_data = test_data.reindex(sorted(test_data.columns), axis=1)

    result = {}
    for i in range(len(test_data)):
        distances = []
        for j in range(len(train_data)):
            distances.append(
                (distance_metric(test_data.iloc[i], train_data.iloc[j]), j)
            )
        sorted_distances = sorted(distances, key=lambda x: x[0])

        k_nearest = sorted_distances[:k]
        k_nearest_labels = [train_data.iloc[x[1]]["Label"] for x in k_nearest]

        result[test_data.iloc[i].name] = max(
            set(k_nearest_labels), key=k_nearest_labels.count
        )

    return result


def niave_bayes(train_data, test_data):
    label_prob = {}
    for label in range(0, 10):
        label_prob[label] = len(train_data[train_data["Label"] == label]) / len(
            train_data
        )

    result = {}
    for row in range(len(test_data)):
        row_max = 0
        row_max_label = None

        prob = {}
        for label in range(0, 10):
            # Calculate probability of label given this row
            # p(label | row) = p(label) * p(row,col | label) for all rows
            for col in train_data.columns[100:]:
                prob[(col, label)] = (
                    len(
                        train_data[
                            (train_data["Label"] == label)
                            & (train_data[col] == test_data.iloc[row][col])
                        ]
                    )
                    + 1
                ) / (
                    len(train_data[train_data["Label"] == label])
                    + len(train_data[col].unique())
                )

        for label in range(0, 10):
            row_prob = label_prob[label]  # p(label)
            for col in train_data.columns[100:]:
                row_prob *= prob[(col, label)]

            if row_prob > row_max:
                row_max = row_prob
                row_max_label = label
        result[test_data.iloc[row].name] = row_max_label
    return result
