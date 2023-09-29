import numpy as np

def cossim(a: np.array, b: np.array):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

