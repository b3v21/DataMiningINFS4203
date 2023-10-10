import pandas as pd
from normalisation import standardise
from metrics import manhattan
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def one_hot_encode(data):
    for col in data.columns[100:]:
        if col == "Label":
            continue
        data = pd.concat(
            [data, pd.get_dummies(data[col], prefix=col).astype(int)], axis=1
        )
        data = data.drop(col, axis=1)
    return data


def k_NN(k, train_data, test_data, distance_metric):
    train_data.drop(train_data.columns[100:128], axis=1, inplace=True)
    test_data.drop(test_data.columns[100:128], axis=1, inplace=True)
    
    neigh = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
    neigh.fit(train_data.drop("Label", axis=1), train_data["Label"])
    
    result = neigh.predict(test_data)
    return result


def niave_bayes_gaussian(train_data, test_data):
    train_data.drop(train_data.columns[100:128], axis=1, inplace=True)
    test_data.drop(test_data.columns[100:128], axis=1, inplace=True)
    
    gnb = GaussianNB()
    gnb.fit(train_data.drop("Label", axis=1), train_data["Label"])
    
    result = gnb.predict(test_data)
    return result

def niave_bayes_multinominal(train_data, test_data):
    train_data.drop(train_data.columns[:100], axis=1, inplace=True)
    test_data.drop(test_data.columns[:100], axis=1, inplace=True)
    
    mnb = MultinomialNB(force_alpha=True)
    mnb.fit(train_data.drop("Label", axis=1), train_data["Label"])
    
    result = mnb.predict(test_data)
    return result
