import numpy as np
a = np.array([1 ,2])
scarlar = np.linalg.norm(a)
print (scarlar)
nml = np.true_divide(a , scarlar)
print (nml)

b = np.array([[1,1]]).T
print (np.linalg.norm(b))