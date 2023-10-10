import numpy as np
import sklearn.metrics as skm

def accuracy (y_true, y_pred):
    return skm.accuracy_score(y_true, y_pred)

def manhattan(p1, p2):
    return sum(np.abs(np.array(p1) - np.array(p2)))


def euclidean(p1, p2):
    return sum(np.square(np.array(p1) - np.array(p2)))


def mc_f1(y_true, y_pred):
    return skm.f1_score(y_true, y_pred, average="macro")
