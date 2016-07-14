import numpy as np


class ArrayFormatter:

    def __init__(self):
        pass

    # shuffle the data order so it's randomized for each training session
    def shuffle(self, a):
        a = np.array(a)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]

        return shuffled_a

    # shuffle the data order so it's randomized for each training session
    def shuffleInUnison(self, a, b):
        assert len(a) == len(b)
        a = np.array(a)
        b = np.array(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

