import numpy as np
from matplotlib import pyplot as plt

x = [1, 2, 3, 4, 5, 7, 9, 10, 12, 13]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
plt.plot(x, y, 'o')
plt.show()
xlist, ylist = [], []
for i in range(len(x)):
    xlist.append([x[i]])
    ylist.append([y[i]])
xlist, ylist = np.mat(xlist, dtype=float), np.mat(ylist, dtype=float)
print(xlist, ylist)
w1 = np.dot(np.dot(np.dot(xlist.T, xlist).I, xlist.T), ylist)
print(w1)
newy = []
for i in x:
    newy.append(np.array(w1 * i)[0][0])
print(newy)
plt.plot(x, y, 'ro')
plt.plot(x, newy)
plt.show()
