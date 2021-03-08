
import numpy as np
from numpy import exp
from scipy.special import factorial
import matplotlib.pyplot as plt

import hw01

def poisson_PDF(k, mu):
    return(mu**k / factorial(k) * exp(-mu))


from hw01 import poisson_MLE

k_range = range(0, 10)
k_range = list(k_range)
data = [2,5,0, 3, 1, 3]

print("input sequence (Poisson Distribution):")
print(data)
pois_MLE = poisson_MLE(data)
print("MLE :")
print(pois_MLE)


fig0 = plt.figure(0)
plt.plot(k_range, data,
            label=f'$\mu$={pois_MLE}',
            #alpha=0.5,
            marker='x',
            markersize=8)
plt.grid()
plt.title('Poisson PDF')
plt.xlabel('k', fontsize=12)
plt.ylabel('$f(k \mid \lambda)$', fontsize=12)
#plt.axis([0, max, 0, max])
plt.legend(fontsize=12)
plt.show()



ref_mus = [1, 2, 5, 7, 10]
fig1 = plt.figure(1)#0, figsize=(12, 8))
k_range = range(0, 20)

for mu in ref_mus:
    distro = []
    for k_i in k_range:
        distro.append(poisson_PDF(k_i, mu))


    plt.plot(k_range,
            distro,
            label=f'$\mu$={mu}',
            alpha=0.5,
            marker='o',
            markersize=8)
plt.grid()
plt.title('Poisson PDF')
plt.xlabel('k', fontsize=12)
plt.ylabel('$f(k \mid \lambda)$', fontsize=12)
#plt.axis([0, max, 0, max])
plt.legend(fontsize=12)
#plt.show()
