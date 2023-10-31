import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

fn = 0
def fignum():
    global fn
    fn +=1
    return fn

# PART A

for i in range(0,5):
    plt.plot(logwave, flux[i, :], label="Galaxy {galaxyindex}".format(galaxyindex = i))
plt.ylabel(r'Flux [$10^{-17}$ \mathrm{erg} s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]', fontsize=14)
plt.xlabel('Wavelength [$A$]', fontsize = 16)
plt.legend()
plt.title('Part A: Spectrum of the First Galaxy', fontsize=16)
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Part B

# Calculate the total flux for each galaxy
flux_sum = np.sum(flux, axis=1)

# Normalize each galaxy's flux by dividing by its total flux
flux_normalized = flux / np.tile(flux_sum, (np.shape(flux)[1], 1)).T

# After checking that the data is properly normalized, you can label the axes
plt.plot(np.sum(flux_normalized, axis=1))
plt.ylim(0, 2)
plt.xlabel('Galaxy Index', fontsize=14)  # Label the x-axis
plt.ylabel('Normalized Total Flux', fontsize=14)  # Label the y-axis
plt.title('Part B: Normalized Total Flux for Each Galaxy', fontsize=16)
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Part C

# Calculate the mean spectrum of the normalized spectra
means_normalized = np.mean(flux_normalized, axis=1)

# Subtract the mean spectrum from the normalized spectra
flux_normalized_0_mean = flux_normalized - np.tile(means_normalized, (np.shape(flux)[1], 1)).T

# Plot the 0-mean normalized flux for the first galaxy
plt.plot(logwave, flux_normalized_0_mean[0, :])
plt.ylabel('Normalized 0-mean Flux', fontsize=16)
plt.xlabel('Wavelength [$A$]', fontsize=16)
plt.title('Part C: 0-Mean Normalized Flux of the First Galaxy', fontsize=16)
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Part D

def sorted_eigs(r, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr=r.T@r
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    eig = eigs[0][arg] # sort eigenvalues
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec
# here I'll use data for only the first 500 galaxies because my computer is slow!
r = flux_normalized_0_mean 
r_subset = r[:500, :]
logwave_subset = logwave
C = r_subset.T@r_subset # correlation matrix, dimension # wavelengths x # wavelengths
# check dimensions of correlation matrix
C.shape
# check dimension of data matrix for 500 galaxies
r_subset.shape

eigvals, eigvecs = sorted_eigs(r_subset, return_eigvalues = True)

for i in range(5):
    plt.plot(logwave_subset, eigvecs[:, i], label=f'Eigenvector {i+1}')
plt.legend()
plt.ylabel('Normalized 0-mean Flux', fontsize=16)
plt.xlabel('Wavelength [$A$]', fontsize=16)
plt.title('Part D: First Five Eigenvectors', fontsize=16)
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Part E

U, S, Vh = np.linalg.svd(r_subset, full_matrices=True)
# rows of Vh are eigenvectors
eigvecs_svd = Vh.T
eigvals_svd = S**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]

[plt.plot(eigvecs_svd[:,i], eigvecs[:,i], 'o')for i in range(500)]
plt.plot(np.linspace(-0.2, 0.2), np.linspace(-0.2, 0.2))
plt.xlabel('SVD eigenvalues', fontsize = 16)
plt.ylabel('Eig eigenvalues', fontsize = 16)
plt.title("Comparing SVD and Eig eigenvalues")
plt.show()

plt.plot(eigvals_svd, eigvals[:500], 'o')
plt.xlabel('SVD eigenvalues', fontsize = 16)
plt.ylabel('Eig eigenvalues', fontsize = 16)
plt.title("Comparing SVD and Eig eigenvalues") # Now eigenvalues match!
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Part G

def PCA(l, r, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """
    eigvector = sorted_eigs(r)
    eigvec=eigvector[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum
# Using only the first eigenvector does not capture the entire signal well!
nc = 1 
plt.plot(logwave_subset, PCA(nc,r_subset)[1,:], label = 'Nc = {Nc}'.format(Nc = nc))
plt.plot(logwave_subset, r_subset[1,:], label = 'original data')

plt.ylabel('Normalized 0-mean flux', fontsize = 16)
plt.xlabel('Wavelength [$A$]', fontsize = 16)
plt.title('Result of PCA Nc = {Nc}'.format(Nc=nc))
plt.legend()
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

nc = 5
plt.plot(logwave_subset, PCA(nc,r_subset)[1,:], label = 'l = {Nc}'.format(Nc = nc))
plt.plot(logwave_subset, r_subset[1,:], label = 'original data')

plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.title('Result of PCA Nc = {Nc}'.format(Nc=nc))
plt.legend()
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# check to make sure that using all eigenvectors reconstructs original signal

nc = 4001
plt.plot(logwave_subset, PCA(nc,r_subset)[1,:], label = 'l = {Nc}'.format(Nc = nc))
plt.plot(logwave_subset, r_subset[1,:], label = 'original data')

plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.title('Result of PCA Nc = {Nc}'.format(Nc=nc))
plt.legend()
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Part H

# Assuming you have the coefficients c0, c1, and c2
c0 = eigvecs[:, 0]  # Coefficient 0
c1 = eigvecs[:, 1]  # Coefficient 1
c2 = eigvecs[:, 2]  # Coefficient 2

# Create the scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(c0, c1, s=10, alpha=0.5)
plt.xlabel('Coefficient 0', fontsize=14)
plt.ylabel('Coefficient 1', fontsize=14)
plt.title('c0 vs c1', fontsize=16)

plt.subplot(1, 2, 2)
plt.scatter(c0, c2, s=10, alpha=0.5)
plt.xlabel('Coefficient 0', fontsize=14)
plt.ylabel('Coefficient 2', fontsize=14)
plt.title('c0 vs c2', fontsize=16)

plt.tight_layout()
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Part I

# Initialize an array to store the squared fractional residuals
squared_residuals = []

# Create an array to store Nc values from 1 to 20
Nc_values = np.arange(1,21)

# Calculate the squared fractional residuals for each Nc
for Nc in Nc_values:
    # Use your original PCA function to obtain the approximate spectra
    approximated_spectra = PCA(Nc, r_subset, project=True)

    # Compute the squared differences between original and approximate spectra
    squared_diff = np.sum((r_subset - approximated_spectra) ** 2)

    # Compute the squared sum of the original spectra
    squared_original_sum = np.sum(r_subset ** 2)

    # Calculate the squared fractional residual
    squared_residual = squared_diff / squared_original_sum

    squared_residuals.append(squared_residual)

# Plot the squared fractional residuals as a function of Nc
plt.plot(Nc_values, squared_residuals, marker='o')
plt.xlabel('Number of Coefficients (Nc)', fontsize=14)
plt.ylabel(r'Flux [$10^{-17}$ \mathrm{erg} s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]^2', fontsize=14)
plt.title('Squared Fractional Residuals vs. Nc', fontsize=16)
plt.grid(True)
plt.savefig("ps-6-figure-{f}.png".format(f = fignum()))
plt.show()

# Print the squared fractional error for Nc = 20
print(f'Squared Fractional Error for Nc = 20: {squared_residuals[-1]:.6f}')