import numpy as np

A = np.array([[1., 1., 0.],
              [0., 1., 0.],
              [0., 2., 0.]])

u, s, vh = np.linalg.svd(A)

print("==== A ====")
print(A)

print("==== u ====")
print(u)

print("==== s ====")
print(s)

print("==== vh ====")
print(vh)

print("==== null ====")
print(A@A.T)
print(np.ones(3)-(A.T@np.linalg.pinv(A@A.T)@A))