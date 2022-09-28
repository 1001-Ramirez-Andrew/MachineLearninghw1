import numpy as np
a = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
my_filter = np.array([True, False, True, False])
print(a[tuple(my_filter)])