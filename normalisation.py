import pandas as pd
import matplotlib.pyplot as plt
from imputise import class_specifc, all_value
from clean import gaussian_outlier_detection

def standardise(data, cols = None):
    if cols is None:
        cols = range(0, len(data.columns))
    
    for col in data.columns[cols]:
        if col == 'Label':
            continue
        data[col] = (data[col] - data[col].mean())/data[col].std()
    return data

def min_max_normalise(data,cols= None):
    if cols is None:
        cols = range(0, len(data.columns))
    
    for col in data.columns[cols]:
        if col == 'Label':
            continue
        data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())
    return data