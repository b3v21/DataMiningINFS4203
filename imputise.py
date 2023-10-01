import pandas as pd

def class_specifc(data):
    for col in data.columns:
        if col == 'Label':
            continue
        data[col].fillna(data.groupby('Label')[col].transform('mean'), inplace=True)
    return data
    
def all_value(data):
    for col in data.columns:
        if col == 'Label':
            continue
        data[col].fillna(data[col].mean(), inplace=True)
    return data
    