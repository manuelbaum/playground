import numpy as np
import matplotlib.pyplot as plt
# trilateration for overconstrained beacon localization (4 or more beacons)
# using gauss newton iteration
# https://en.wikipedia.org/wiki/Trilateration


# generate ground truth
xs_beacon = np.array([[0,0,0],
                      [0,0,1],
                      [4,0,0],
                      [0,4,0]])
                      
x_gt = np.array([1., 1., 1.])
print(x_gt - xs_beacon)
z = np.linalg.norm(x_gt - xs_beacon, axis=1)

print('z', z)

residuals = []
beta = np.array([0.2,0.1,0.1])
for i in range(30):
    print('best guess:',beta)
    
    # compute jacobian
    z_pred = np.linalg.norm((beta-xs_beacon), axis=1)
    r = (z - z_pred)

    ####################### jacobian not computed correctly right now ####################################
    J = -2 * (beta-xs_beacon)

    delta = -np.linalg.pinv(J).dot(r)
    beta += delta

    residuals.append(r)

plt.plot(residuals)
plt.show()