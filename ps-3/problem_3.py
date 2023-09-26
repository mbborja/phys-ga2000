from random import random
import numpy as np
import matplotlib.pyplot as plt

# Running Tallys
Bi209 = 0
Pb209 = 0
Ti209 = 0
Bi213 = 10000

# Formula for probabilty of decay in 1 step is
# 1 - 2**(-h/tau) where h is the size of time step in seconds
# and tau is the half-life of the isotope

h = 1 # given

prob_Pb209 = 1 - 2**(-h/(3.3*60))
prob_Ti209 = 1 - 2**(-h/(2.2*60))
prob_Bi213 = 1 - 2**(-h/(46*60))

Bi209points = []
Pb209points = []
Ti209points = []
Bi213points = []

t = range(0,20000,h)
for i in t:
	Bi209points.append(Bi209)
	Pb209points.append(Pb209)
	Ti209points.append(Ti209)
	Bi213points.append(Bi213)
	
    #Calculate the number of atoms that decay
	for i in range(Pb209):
		if random()<prob_Pb209:
			Pb209-=1
			Bi209+=1
	
	for i in range(Ti209):
		if random()<prob_Ti209:
			Ti209-=1
			Pb209+=1
	
	for i in range(Bi213):
		if random()<prob_Bi213:
			Bi213 -=1
			if random()<0.9791:
				Pb209+=1
			else:
				Ti209+=1

# Make the graph
plt.plot(t,Bi209points,label='Bi209')
plt.plot(t,Pb209points,label='Pb209')
plt.plot(t,Ti209points,label='Ti209')
plt.plot(t,Bi213points,label='Bi213')
plt.title("Decay of 10000 Bi213 Atoms")
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Number of atoms (N)')
plt.savefig("problem3.png")
plt.show()