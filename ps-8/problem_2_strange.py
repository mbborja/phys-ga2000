import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lorenz equations
def lorenz_equations(t, xyz, sigma, r, b):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

def numerical_traj_ex(t_span, y0, t, sigma, r, b):
    # Define the derivative function for the Lorenz equations
    def derivative_func(t, y):
        return lorenz_equations(t, y, sigma, r, b)

    # Integrate the equations of motion
    sol = solve_ivp(derivative_func, t_span, y0, t_eval=t, method='LSODA')

    # Extract x and z values
    x_result = sol.y[0, :]  # Extracting the first component (x)
    z_result = sol.y[2, :]  # Extracting the third component (z)

    return x_result, z_result

# Set the parameters
sigma = 10
r = 28
b = 8/3

# Set the initial conditions
initial_conditions = [0, 1, 0]

# Set the time span
time_span = (0, 50)

# Set the time points where you want to evaluate the solution
time_points = np.linspace(time_span[0], time_span[1], 10000)

# Solve the Lorenz equations
x_values, z_values = numerical_traj_ex(time_span, initial_conditions, time_points, sigma, r, b)

# Plot z against x
plt.plot(x_values, z_values, label='Lorenz Attractor')
plt.title('Lorenz Attractor: z against x')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.savefig("butterfly.png")
plt.show()
