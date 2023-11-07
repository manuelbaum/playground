import numpy as np

import cv2

def cross_approximate(mat, rows_indices, cols_indices):
    rows = mat[rows_indices, :]
    cols = mat[cols_indices, :]

    mid = np.zeros((len(rows_indices), len(rows_indices)))
    for i in range(len(rows_indices)):
        for j in range(len(cols_indices)):
            mid[i,j] = mat[rows_indices[i],cols_indices[j]]

    mid = np.linalg.inv(mid)
    print(cols.shape, mid.shape, rows.shape)
    X = cols.T@mid@rows
    print(X.shape)
    X = X / np.max(X)
    return X


img = cv2.imread("/home/manuelbaum/Downloads/cat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
size = (np.array(img.shape) / 2).astype(int)
img = cv2.resize(img, tuple(size))

print(img.shape)

cv2.imshow("orginal", (img/np.max(img)).astype(int))
# recon = cross_approximate(img, np.array([100]), np.array([100]))
recon = cross_approximate(img, np.arange(0, 600, 2), np.arange(0, 600, 2))
cv2.imshow("recon", recon)

cv2.waitKey()
