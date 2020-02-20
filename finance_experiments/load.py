
import numpy as np
import pandas as pd
import torch

YEARS = [2014, 2015, 2016, 2017, 2018]

class ToNum:
    def __init__(self):
        self.symbols = []

    def convert(self, symbol):
        if not symbol in self.symbols:
            self.symbols.append(symbol)
            return len(self.symbols) - 1
        else:
            return self.symbols.index(symbol)

    def index(self, key):
        return self.symbols.index(key)


def get_years(years=[]):
    data = [pd.read_csv('data/%s_clean.csv' % y, index_col=0) for y in YEARS]

    k = data[0].keys()
    for d in data:
        k = k.intersection(d.keys())
    print(k)

    data = [pd.read_csv('data/%s_clean.csv' % y, index_col=0) for y in years]
    data = [d[k] for d in data]

    k_data = k.drop(['Sector', 'Class'])
    k_target = 'Class'

    _data = []
    for d in data:
        num_0 = d['Class'].isin([0]).sum()
        num_1 = d['Class'].isin([1]).sum()
        n = min(num_0, num_1)
        class_0 = d.nsmallest(n, 'Class')
        class_1 = d.nlargest(n, 'Class')
        _data.append(class_0.append(class_1).sample(frac=1))
    data = _data

    x = [d[k_data].to_numpy() for d in data]
    y = [d[k_target].to_numpy() for d in data]

    return [{
        'images': torch.tensor(x_, dtype=torch.float32).cuda(),
        'labels': torch.tensor(y_.reshape(-1, 1), dtype=torch.float32).cuda(),
        'info': yr_,
        } for x_, y_, yr_ in zip(x, y, years)]


def get_sectors():
    data = [pd.read_csv('data/%s_clean.csv' % y, index_col=0) for y in YEARS]

    k = data[0].keys()
    for d in data:
        k = k.intersection(d.keys())

    data = [d[k] for d in data]

    # stack everything
    _data = None
    for d in data:
        if _data is None:
            _data = d
        else:
            _data = _data.append(d)
    data = _data

    # sort by sectors
    sectors = data['Sector'].unique()
    _data = []
    for s in sectors:
        ids = data['Sector'].isin([s])
        _data.append(data[ids].sample(frac=1))
    data = _data

    _data = []
    for d in data:
        num_0 = d['Class'].isin([0]).sum()
        num_1 = d['Class'].isin([1]).sum()
        n = min(num_0, num_1)
        class_0 = d.nsmallest(n, 'Class')
        class_1 = d.nlargest(n, 'Class')
        _data.append(class_0.append(class_1).sample(frac=1))
    data = _data

    k_data = k.drop(['Sector', 'Class'])
    k_target = 'Class'

    x = [d[k_data].to_numpy() for d in data]
    y = [d[k_target].to_numpy() for d in data]

    return [{
        'images': torch.tensor(x_, dtype=torch.float32).cuda(),
        'labels': torch.tensor(y_.reshape(-1, 1), dtype=torch.float32).cuda(),
        'info': s_
        } for x_, y_, s_ in zip(x, y, sectors)]


if __name__ == '__main__':

    x, y = get_envs()
