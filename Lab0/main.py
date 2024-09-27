import time
import pandas as pd
import numpy as np
def sum_numpy():
    start = time.time()
    X = np.arange(1000000)
    Y = np.arange(1000000)
    Z=X+Y
    return time.time() - start
print("time sum:" + str(sum_numpy()))

arr = np.array([1, 2, 3], float)
arr.tolist()
list(arr)
print(str(arr))

arr2 = np.random.normal(0,1,5)
print(str(arr2))

arr3 = np.identity(5, dtype=float)
print(str(arr3))

arr4 = np.eye(3, k=1, dtype=float)
print(str(arr4))
