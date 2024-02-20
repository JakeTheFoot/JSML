import numpy as np
dvalues = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.pad(np.rot90(np.rot90(dvalues)), 1, mode='constant'))