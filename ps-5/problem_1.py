import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp

# Define the range of x values from 0 to 5
x = np.linspace(0, 5, 1000)

# Define values of 'a' for the three curves
a_values = [2, 3, 4]

# Create a plot for each value of 'a'
for a in a_values:
    # Calculate the gamma function for each x
    gamma_values = x**(a - 1) * np.exp(-x)
    
    # Plot the curve
    plt.plot(x, gamma_values, label=f'a = {a}')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('Integrand')
plt.title('Integrand for Different Values of a')
plt.legend()
plt.savefig("1a.png")
# Show the plot
plt.grid()
plt.show()

# Part e)

def gamma(a):
    # Define the integrand function with the new expression
    def integrand(z):
        c = a - 1
        return np.exp((a - 1) * (np.log(c * z) - np.log(1 - z)) - (c * z) / (1 - z)) * c / (1 - z)**2
    
    # Integrate the function using quad from SciPy
    result, _ = sp.quad(integrand, 0, 1)
    
    return result

# Test the gamma function for a specific value (e.g., a = 10)
result = gamma(3/2)

print("Gamma(3/2) =", result)
print("1/2 (pi)^1/2 is approx ", 0.886)
print()
# Part f) 

from math import factorial

test_list = [3,6,10]

print("Testing gamma(a) vs factorial(a-1)")
for a in test_list:
    print("Prebuilt Gamma function gamma({Value}): \t".format(Value = a), gamma(a))
    print("Factorial({Value}): \t \t \t \t".format(Value = a-1), factorial(a-1))
    print()
    
