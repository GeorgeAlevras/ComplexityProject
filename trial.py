import numpy as np
from collections import Counter


a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1], [1, 1, 1], [2, 1, 1], [2, 2, 1]])
b = Counter(map(tuple, a))

res = [k for k in b.keys() if b[k]>1]

print(len(res))