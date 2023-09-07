import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian parameters
mean = 0
std_dev = 3

# Generate x values automatically based on data
x = np.linspace(mean - 5 * std_dev, mean + 5 * std_dev, 1000)

# Calculate the Gaussian values
y = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

# Normalize the Gaussian curve
area_under_curve = np.trapz(y, x)
y /= area_under_curve

# Create the plot
plt.plot(x, y, label=f'Gaussian (μ={mean}, σ={std_dev})')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.title('Normalized Gaussian Distribution')
plt.legend()

# Save the plot as a PNG file
plt.savefig('gaussian.png')

# Display the plot (optional)
plt.show()