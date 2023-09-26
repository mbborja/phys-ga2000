import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0  # Mass
w = 1.0  # Omega

# Initial conditions
x0 = 1.0  # Initial position
p0 = 1.0  # Initial momentum

# Time parameters
t_max = 20.0  # Maximum simulation time
dt = 0.01    # Time step

# Initialize arrays to store position and momentum values
num_steps = int(t_max / dt) + 1
x_values = np.zeros(num_steps)
p_values = np.zeros(num_steps)

# Initial conditions
x_values[0] = x0
p_values[0] = p0

# Perform numerical simulation using the Euler method
for i in range(1, num_steps):
    # Update momentum and position using Hamilton's equations
    p_values[i] = (m*w*x0*np.sinh(w*(dt*i))) - (p0 * np.cosh(w*(dt*i)))
    x_values[i] = (x0*np.cosh(w*(dt*i))) - (p0/(m*w) * np.sinh(w*(dt*i)))
# Plot the phase-space trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_values, p_values, label='Phase-space trajectory')
plt.xlabel('Position (x)')
plt.ylabel('Momentum (p)')
plt.title('Phase-space Trajectory of the Inverted Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()