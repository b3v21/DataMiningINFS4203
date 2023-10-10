import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from scipy import stats


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

def niave_bayes_multinominal(train_data, test_data):
    train_data.drop(train_data.columns[:100], axis=1, inplace=True)
    test_data.drop(test_data.columns[:100], axis=1, inplace=True)
    
    mnb = MultinomialNB(force_alpha=True)
    mnb.fit(train_data.drop("Label", axis=1), train_data["Label"])
    
    result = mnb.predict(test_data)
    return result

def decision_tree(train_data, test_data):
    train_data.drop(train_data.columns[:100], axis=1, inplace=True)
    test_data.drop(test_data.columns[:100], axis=1, inplace=True)
    
    clf = DecisionTreeClassifier()
    clf.fit(train_data.drop("Label", axis=1), train_data["Label"])
    
    result = clf.predict(test_data)
    return result

def ensemble_kNN_nb_dt(k, train_data, test_data, dist_metric):

    kNN_res = k_NN(k, train_data.copy(), test_data.copy(), dist_metric)
    decision_tree_res = decision_tree(train_data.copy(), test_data.copy())
    niave_bayes_multinominal_res = niave_bayes_multinominal(train_data.copy(), test_data.copy())
    
    res = [stats.mode(x)[0] for x in zip(kNN_res, decision_tree_res, niave_bayes_multinominal_res)]
    return np.array(res).flatten()