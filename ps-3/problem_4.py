import numpy as np
import matplotlib.pyplot as plt
from random import random

N = 1000
tau = 3.053*60
mu = np.log(2)/tau

z = np.random.rand(N)

t_dec = -1/mu*np.log(1-z)
t_dec = np.sort(t_dec)
decayed = np.arange(1,N+1)
surived = N - decayed

plt.plot(t_dec,surived, label="Survived")
plt.plot(t_dec,decayed, label = "Decayed")
plt.plot(t_dec,t_dec)
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Number (N)")
plt.title("Radioactive Decay of 1000 Ti208")
#plt.savefig("problem4.png")
plt.show()