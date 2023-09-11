import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the Gaussian Function
mean = np.float64(0)
std_dev = np.float64(3)

# Define an ndarray which contains the mean and 5 standard deviations away from it
x = np.linspace(mean - 5 * std_dev, mean + 5 * std_dev, 1000)

# Standard Gaussian Method with mean and standard deviation
def Gaussian(x, mean, std_dev):
    """ Returns the evaluation of a standard gaussian function given the supplied mean and standard deviation
    
    Args:
        x (ndarray): linspace 
        mean (float64): The center or mean value of the gaussian
        std_dev (float64): Standard Deviation

    Returns:
        _type_: _description_
    """
    y = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return y

y = Gaussian(x, mean, std_dev)

# This normalizes the gaussian curve
area_under_curve = np.trapz(y, x) # Integrates the full area under the curve given the x and y values
y /= area_under_curve # Divides each value of y by the value of area_under_curve

# Create the plot
plt.plot(x, y, label=f'Gaussian (μ={mean}, σ={std_dev})')
plt.xlabel('X-Value')
plt.ylabel('Probability Density')
plt.title('Normalized Gaussian Distribution')
plt.legend()

# Save the plot as a PNG file
plt.savefig('gaussian.png')