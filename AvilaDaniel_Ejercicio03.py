import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

data = np.loadtxt("notas_andes.dat.txt")
data[:4,:]

Y = data[:,4]
X = data[:,:4]

b1 = np.ones(2000)
b2 = np.ones(2000)
b3 = np.ones(2000)
b4 = np.ones(2000)

for i in np.arange(2000):
    indices = np.random.randint(0,len(Y),69)
    regresion = sklearn.linear_model.LinearRegression()
    regresion.fit(X[indices], Y[indices])
    b1[i] = regresion.coef_[0]
    b2[i] = regresion.coef_[1]
    b3[i] = regresion.coef_[2]
    b4[i] = regresion.coef_[3]

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
plt.hist(b1, color='g', bins=20)
plt.title("b1 = "+ '{:.2f}'.format(b1.mean()) + " $\pm$ " + '{:.2f}'.format(b1.std()))

plt.subplot(2,2,2)
plt.hist(b2,color='c',bins=20)
plt.title("b1 = "+ '{:.2f}'.format(b2.mean()) + " $\pm$ " + '{:.2f}'.format(b2.std()))

plt.subplot(2,2,3)
plt.hist(b3, bins=20)
plt.title("b1 = "+ '{:.2f}'.format(b3.mean()) + " $\pm$ " + '{:.2f}'.format(b3.std()))

plt.subplot(2,2,4)
plt.hist(b4, color='m',bins=20)
plt.title("b1 = "+ '{:.2f}'.format(b4.mean()) + " $\pm$ " + '{:.2f}'.format(b4.std()))
plt.show()
plt.savefig("bootstrap.png")