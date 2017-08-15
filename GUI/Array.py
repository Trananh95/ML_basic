import numpy as np
import scipy
np1 = np.array([[1,2],[2,3]])
np2 = np.array([[1,1],[1,1]])
# result va vector la cac vector sau khi duoc trai phang
A = np1.view(np.ndarray)
A.shape = -1
result = np.array(np1).flatten()

#random la ma tran hop cac vector lai, chieu la so chieu vector * so vector
random = np.concatenate([[A],[result]])
random = np.concatenate([random, [A]])
random = scipy.delete(random, 0, 0)  # delete second row of A
print (random)