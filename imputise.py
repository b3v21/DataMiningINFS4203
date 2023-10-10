import pandas as pd
import operator
import numpy as np

def custom_mean_class_spec(series, data, col):
    non_nan_values = series.dropna()
    if non_nan_values.empty:
        return data[col].mean()  # Custom value when no non-null values
    else:
        return non_nan_values.mean()
    
def custom_mean_all_val(series):
    non_nan_values = series.dropna()
    if non_nan_values.empty:
        return 0  # Custom value when no non-null values
    else:
        return non_nan_values.mean()

def class_specific(data):
    for col in data.columns[0:100]:
        data = data.replace("nan", pd.NA)
        data[col].fillna(data.groupby("Label")[col].transform(custom_mean_class_spec, data, col), inplace=True)
    for col in data.columns[100:]:
        if col == "Label":
            continue
        freq_dict = dict(
            data.groupby("Label")[col].agg(
                lambda x: pd.Series.mode(x).iat[0] if len(pd.Series.mode(x)) > 0 else 0
            )
        )

        keyMax = max(freq_dict.items(), key=operator.itemgetter(1))[0]
        data[col].fillna(keyMax, inplace=True)

    return data


def all_value(data):
    for col in data.columns[0:100]:
        data = data.replace("nan", pd.NA)
        data[col].fillna(custom_mean_all_val(data[col]), inplace=True)
    for col in data.columns[100:]:
        if col == "Label":
            continue

        mode = data[col].mode()[0]
        if isinstance(mode, np.ndarray):
            mode = mode[0]
        
        if mode == "nan":
            mode = np.random.choice(data[col].unique())

        data[col].fillna(mode, inplace=True)
    return data
