from sklearn.ensemble import IsolationForest


def gaussian_outlier_detection(data):
    for col in data.columns[:100]:
        if col == "Label":
            continue
        mean = data[col].mean()
        std = data[col].std()
        data = data[data[col] < mean + 2 * std]
        data = data[data[col] > mean - 2 * std]
    return data

def isolation_forest(data):
    clf = IsolationForest(contamination=0.05)
    clf.fit(data)
    is_outlier = clf.predict(data) == -1
    
    outlier_indices = data.index[is_outlier]
    data.drop(outlier_indices, inplace=True)
    
    return data

