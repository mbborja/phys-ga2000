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

    # Extract time and y values
    t_result = sol.t
    y_result = sol.y[1, :]  # Extracting the second component (y)

    return t_result, y_result

# Set the parameters
sigma = 10
r = 28
b = 8/3

# Set the initial conditions
initial_conditions = [0, 1, 0]

# Set the time span
time_span = (0, 50)

# Set the time points where you want to evaluate the solution
time_points = np.linspace(time_span[0], time_span[1], 1000)

# Solve the Lorenz equations
t_values, y_values = numerical_traj_ex(time_span, initial_conditions, time_points, sigma, r, b)

# Plot y as a function of time
plt.plot(t_values, y_values, label='y(t)')
plt.title('Lorenz Equations: y as a function of time')
plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.savefig("lorenz.png")
plt.show()