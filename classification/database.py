import os
import numpy as np
from scipy.spatial.distance import cdist

__dirname = os.path.dirname(__file__)
_DATABASE_PATH = os.path.join(__dirname, './database.npy')

if os.path.isfile(_DATABASE_PATH):
    db = np.load(_DATABASE_PATH)
else:
    db = None

def store(data):
    db = np.array(data)
    np.save(_DATABASE_PATH, db)


def topN(vec, n=5, method='cosine'):
    if db is None:
        raise Exception('empyt db')
    result = cdist([vec], db, metric=method)[0]
    indices = np.argsort(result)[:n]
    return indices
