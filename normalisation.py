
def standardise(data, cols=None):
    if cols is None:
        cols = range(0, len(data.columns))

    for col in data.columns[cols]:
        if col == "Label":
            continue
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data


def min_max_normalise(data, cols=None):
    if cols is None:
        cols = range(0, len(data.columns))

    for col in data.columns[cols]:
        if col == "Label":
            continue
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data
