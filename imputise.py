import pandas as pd
import operator
import numpy as np

def class_specifc(data):
    for col in data.columns[0:100]:
        data = data.replace('nan', pd.NA)
        data[col].fillna(data.groupby('Label')[col].transform('mean'), inplace=True)
    for col in data.columns[100:]:
        if col == 'Label':
            continue
        freq_dict = dict(data.groupby('Label')[col].agg(lambda x: pd.Series.mode(x).iat[0] if len(pd.Series.mode(x)) > 0 else 0))
        
        keyMax = max(freq_dict.items(), key = operator.itemgetter(1))[0]
        data[col].fillna(keyMax, inplace=True)

    return data
    
def all_value(data):
    for col in data.columns[0:100]:
        data = data.replace('nan', pd.NA)
        data[col].fillna(data[col].mean(), inplace=True)
    for col in data.columns[100:]:
        if col == 'Label':
            continue
        
        mode = data[col].mode()
        if isinstance(mode, np.ndarray):
            mode = mode[0]
        
        data[col].fillna(mode, inplace=True)
    return data
    