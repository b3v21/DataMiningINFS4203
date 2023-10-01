import pandas as pd
import matplotlib.pyplot as plt
from imputise import class_specifc, all_value
from clean import gaussian_outlier_detection

def standardise(data):
    for col in data.columns:
        if col == 'Label':
            continue
        data[col] = (data[col] - data[col].mean())/data[col].std()
    return data