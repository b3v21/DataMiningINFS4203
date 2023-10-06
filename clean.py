import pandas as pd
from imputise import class_specifc, all_value
from scipy.spatial.distance import pdist, squareform


def gaussian_outlier_detection(data):
    for col in data.columns[:100]:
        if col == "Label":
            continue
        mean = data[col].mean()
        std = data[col].std()
        data = data[data[col] < mean + 2 * std]
        data = data[data[col] > mean - 2 * std]
    return data


data = pd.read_csv("data/train.csv")
data = gaussian_outlier_detection(all_value(data))
