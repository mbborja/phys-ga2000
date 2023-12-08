import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.constants import hbar, electron_mass

# Constants
L = 1e-8
N = 1000
dx = L / N
dt = 1e-18 * 10
x_values = np.linspace(0, L, N)

# Crank-Nicolson coefficients
a1 = 1 + (hbar * dt) / (2 * electron_mass * dx**2) * 1j
a2 = -hbar * dt / (4 * electron_mass * dx**2) * 1j
b1 = 1 - (hbar * dt) / (2 * electron_mass * dx**2) * 1j
b2 = hbar * dt / (4 * electron_mass * dx**2) * 1j

# Initial wavefunction
def psi0(x):
    x0 = L / 2
    sigma = 1e-10
    k = 5e10
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k * x)

real_values = []

# Crank-Nicolson time evolution
psi = psi0(x_values)


for _ in range(100):
    psi_new = np.zeros(N, dtype=np.complex128)
    psi_new[1:-1] = a1 * psi[1:-1] + a2 * (psi[:-2] + psi[2:]) + b1 * psi_new[1:-1] + b2 * (psi_new[:-2] + psi_new[2:])
    
    # Update for the next iteration
    psi = psi_new.copy()
    
    # Append the current state to the list for visualization
    real_values.append(np.real(psi))

# Animation
fig, ax = plt.subplots()
line, = ax.plot(x_values, real_values[0])
annotation = ax.text(0.75, 0.9, f'Time Step: 0', transform=ax.transAxes, color='red')


plt.xlabel('Position (m)')
plt.ylabel('Real Part of Wave Function')
def update(frame):
    line.set_ydata(real_values[frame])
    annotation.set_text('Time Step: {frame}'.format(frame = frame))
    return line,

ani = FuncAnimation(fig, update, frames=len(real_values), interval=50, blit=True)

# For inline animations in Jupyter Notebook
from IPython.display import HTML
HTML(ani.to_jshtml())
ani.save('quantum_wavefunction_evolution.gif', writer='imagemagick')
