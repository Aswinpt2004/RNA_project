import numpy as np

def one_hot_encode_sequence(seq):
    map = {
        'A': [1, 0, 0, 0],
        'U': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'C': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25]
    }
    return np.array([map.get(base.upper(), map['N']) for base in seq])
