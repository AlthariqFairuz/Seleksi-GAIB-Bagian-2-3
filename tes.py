import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)

b = np.expand_dims(a, axis=0)
print(b.shape)
print(b)

c = np.expand_dims(a, axis=1)
print(c.shape)
print(c)