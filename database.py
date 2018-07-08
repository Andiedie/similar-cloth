import os
import pandas
import numpy as np
__dirname = os.path.dirname(__file__)
_DATABASE_PATH = os.path.join(__dirname, './database.csv')

if not os.path.isfile(_DATABASE_PATH):
    pandas.DataFrame(data=[], columns=['key', 'value']).to_csv(_DATABASE_PATH, index=False)
df = pandas.read_csv(_DATABASE_PATH)
dirty = False

def store(key, vec):
    global dirty
    pandas.DataFrame(data=[{
        'key': key,
        'value': ','.join(str(x) for x in vec)
    }]).to_csv(_DATABASE_PATH, mode='a', header=False, index=False)
    dirty = True


def _cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def _euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def topN(vec, n=5, method='cos'):
    global dirty, df
    if dirty:
        df = pandas.read_csv(_DATABASE_PATH)
        dirty = False
    keys = []
    values = []
    for _, row in df.iterrows():
        temp = np.array([float(x) for x in row['value'].split(',')])
        if method == 'cos':
            value = _cosine_similarity(vec, temp)
        elif method == 'euc':
            value = -1 * _euclidean_distance(vec, temp)
        else:
            raise Exception('Unknown Method')
        keys.append(row['key'])
        values.append(value)
    indices = np.argsort(values)[-n:][::-1]
    return np.array(keys)[indices]
