import numpy as np
import matplotlib.pyplot as plt
from random import random

NT1 = 1000
NPb = 0
tau = 3.053*60
h = 1
p = 1- 2**(-h/tau)
tmax = 1000
newprob = np.log(2)/tau

tpoints = np.arange(0.0, tmax, h)
T1points = []
Pbpoints = []

equation  = 2


N = 1000
tau = 3.053*60
mu = np.log(2)/tau

z = np.random.rand(N)

t_dec = -1/mu*np.log(1-z)
t_dec = np.sort(t_dec)
decayed = np.arange(1,N+1)

surived = -decayed + N

plt.plot(t_dec,surived, label="Survived")
plt.plot(t_dec,decayed, label = "Decayed")
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Number (N)")
plt.title("Radioactive Decay of 1000 Ti208")
plt.show()
plt.savefig("problem4.png")