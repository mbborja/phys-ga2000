import numpy as np
import matplotlib.pyplot as plt

# Part A) Plot the data. -----------------------------

# Create empty lists to store the time and signal values
time_values = []
signal_values = []

# Open the data file and read line by line
with open("signal.dat", "r") as file:
    next(file)  # Skip the header line
    for line in file:
        # Split each line into time and signal
        parts = line.split("|")
        time = float(parts[1].strip())
        signal = float(parts[2].strip())
        
        # Append the values to the respective lists
        time_values.append(time)
        signal_values.append(signal)

# Convert lists to NumPy arrays for easier manipulation
time = np.array(time_values)
signal = np.array(signal_values)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, label="Signal Data")
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.title("Signal Data vs. Time")
plt.legend()
plt.grid(True)
plt.savefig("2.png")
plt.show()

# PART B) Use the SVD Technique -----------------------

# Sort the data by time
sorted_indices = np.argsort(time)
time = time[sorted_indices]
signal = signal[sorted_indices]
timep = (time - time.mean()) / time.std() # To avoid round off error we shift time around its mean

order = 3

def polynomial_fit(order, signal, time):

    print("Running polynomial SVD fit with order: {o}".format(o = order))
    
    # Create the design matrix for the higher-order polynomial
    A = np.column_stack([timep**i for i in range(order + 1)])
    
    # Perform Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # Calculate the coefficients of the higher-order polynomial fit
    coefficients = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ signal

    # Generate the higher-order polynomial fit
    poly_fit = A @ coefficients

    # Plot the data and the higher-order polynomial fit
    plt.figure(figsize=(10, 6))
    plt.scatter(time, signal, label="Signal Data")
    plt.plot(time, poly_fit, label=f"{order}-th Order Polynomial Fit", color='red')
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.title(f"{order}-th Order Polynomial Fit (Sorted Data)")
    plt.legend()
    plt.grid(True)
    plt.savefig("2b_order_{o}.png".format(o = order))
    plt.show()
    
    return(poly_fit, coefficients)

order_3_SVD_ym, order_3_SVD_c = polynomial_fit(order, signal, timep)

# Part C) Calculate the Residuals ---------------------------------

uncertainty = 2

def reduced_chi_squared(uncertainty, ym, signal, c):
    """_summary_

    Args:
        uncertainty (_type_): _description_
        ym (_type_): _description_
        signal (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    residuals = signal - ym
    chi_squared = np.sum((residuals/uncertainty)**2)
    degrees_of_freedom = len(signal) - len(c)
    reduced_chi_squared = chi_squared / degrees_of_freedom
    return reduced_chi_squared

print("Polynomial Order 3 SVD Reduced Chi Squared: ", reduced_chi_squared(uncertainty, order_3_SVD_ym, signal, order_3_SVD_c))

# Part D) Try a much higher order polynomial.

order_30_SVD_ym, order_30_SVD_c = polynomial_fit(30, signal, timep)

print("Polynomial Order 30 SVD Reduced Chi Squared: ", reduced_chi_squared(uncertainty, order_30_SVD_ym, signal, order_30_SVD_c))

# Part E) Try fitting a set of sin and cos functions plus a zero-point offset.

#estimate linear trend
N = len(time)
A = np.zeros((len(time), 2))
A[:, 0] = 1.
A[:, 1] = time
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c)

#subtract off linear trend
flat_signal = signal-ym

# Number of test offsets
testN = 100
offsets = np.linspace(0, np.pi, testN)

# Initialize arrays to store results
best_offset = None
best_A = None
best_B = None
best_ym = None
best_reduced_chi_squared = float('inf')
amplitude = 0.28

# calculate FFT of oscillations
from scipy.fft import fft, fftfreq
# sample spacing
T = time[1]-time[0]
yf = fft(flat_signal)
xf = fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), '-o')
plt.grid()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Fast Fourier Transform (FFT) of Oscillations")
omega = 2*np.pi*xf[np.argsort(2.0/N * np.abs(yf[0:N//2]))[::-1]][0]
plt.vlines(omega/(2*np.pi), 0, 2, color = 'red')
plt.savefig("2e.png")
plt.show()

for offset in offsets:
    A = np.zeros((len(time), 4))
    A[:, 0] = 1.
    A[:, 1] = time
    A[:, 2] = amplitude*(np.cos(omega * time + offset))
    A[:, 3] = amplitude*(np.sin(omega * time + offset))

    # Perform SVD
    u, w, vt = np.linalg.svd(A, full_matrices=False)
    ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    c = ainv.dot(signal)
    
    # Extract coefficients for amplitudes
    A_fit = c[2]  # Amplitude for the cosine term
    B_fit = c[3]  # Amplitude for the sine term
    
    # Update the design matrix with the optimized amplitudes
    A[:, 2] = A_fit * np.cos(omega * time)
    A[:, 3] = B_fit * np.sin(omega * time)
    
    ym = A.dot(c)

    rcs = reduced_chi_squared(2,ym,signal,c)
    # # Calculate the residuals
    # residuals = signal - ym
    # # Calculate the reduced chi-squared value
    # uncertainty = 2.0  # Standard deviation of measurement uncertainties
    # chi_squared = np.sum((residuals / uncertainty)**2)
    # degrees_of_freedom = len(signal) - len(c)
    # reduced_chi_squared = chi_squared / degrees_of_freedom

    
    
    # Store the best model based on the lowest reduced chi-squared
    if rcs < best_reduced_chi_squared:

        best_reduced_chi_squared = rcs
        best_offset = offset
        best_A = A_fit
        best_B = B_fit
        best_ym = ym
    
plt.plot(time, signal, '.', label='data')
plt.plot(time, best_ym, label='best model (offset={:.2f}, A={:.2f}, B={:.2f})'.format(best_offset, best_A, best_B))
plt.xlabel('t/t_max')
plt.ylabel('Signal Value')
plt.title("Best fit model using set of sin and cos functions")
plt.legend()
plt.savefig("2e_2.png")
plt.show()

print("Best Reduced_Chi_Squared: ", best_reduced_chi_squared)
print("Best Offset:", best_offset)
print("Best A:", best_A)
print("Best B:", best_B)