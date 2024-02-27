import numpy as np

# xs are known landmark positions
# p is ground truth object position
# ds are distances between object and landmarks

# 1d
xs = np.array([[-2.],[ -1],[ 2.]])
p = np.array([0.5])
ds = xs-p
noise = np.random.normal(0, .5, 3)
print('noise', noise)

ds += np.expand_dims(noise, 1)

print(ds)

A = np.ones((3,1))
b = xs-ds

a_pinv = np.linalg.pinv(A)
x = a_pinv.dot(b)

print('a', A)
print('a_pinv', a_pinv)

print(x)
