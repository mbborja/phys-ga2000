import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

def p(x, beta_0, beta_1):
    return 1 / (1 + np.exp(-(beta_0 + beta_1 * x)))

x = 50

beta_0 = np.linspace(-5, 5, 100)
beta_1 = np.linspace(-5, 5, 100)
beta = np.meshgrid(beta_0, beta_1)
p_grid = p(x, *beta)
plt.pcolormesh(*beta, p_grid)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize=16)
plt.ylabel(r'$\beta_1$', fontsize=16)
plt.title(r'$p(y_i|x_i=50,\beta_0, \beta_1)$', fontsize=16)
plt.savefig("p2.png")
plt.show()

data = pd.read_csv('survey.csv')

# send data to numpy bc pandas annoys me
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
xs = xs[x_sort]
ys = ys[x_sort]

def neg_log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    ll_list = [ys[i] * np.log(p(xs[i], beta_0, beta_1) + epsilon) +
               (1 - ys[i]) * np.log(1 - p(xs[i], beta_0, beta_1) + epsilon) for i in range(len(xs))]
    neg_ll = -np.sum(np.array(ll_list), axis=-1)
    return neg_ll

# Visualizing negative log-likelihood
beta_0 = np.linspace(-5, 5, 100)
beta_1 = np.linspace(-5, 5, 100)
beta_mesh = np.meshgrid(beta_0, beta_1)
neg_ll_mesh = neg_log_likelihood(beta_mesh, xs, ys)
plt.pcolormesh(*beta_mesh, neg_ll_mesh)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize=16)
plt.ylabel(r'$\beta_1$', fontsize=16)
plt.title(r'$-\mathcal{L}(\beta_0, \beta_1)$', fontsize=16)
plt.savefig("p2_2.png")
plt.show()

# Initial parameter values
beta_start = [0, 0]

# Minimizing negative log-likelihood
result = optimize.minimize(neg_log_likelihood, beta_start, args=(xs, ys))

# Hessian (inverse) matrix and variance of the residuals
hess_inv = result.hess_inv
var_residuals = result.fun / (len(ys) - len(beta_start))

# Covariance matrix of parameters
def covariance(hess_inv, var_residuals):
    return hess_inv * var_residuals

# Error of parameters
def parameter_error(hess_inv, var_residuals):
    cov_matrix = covariance(hess_inv, var_residuals)
    return np.sqrt(np.diag(cov_matrix))

# Printing results
print('Optimal parameters and errors:')
print(f'\tbeta_0: {result.x[0]:.4f} +/- {parameter_error(hess_inv, var_residuals)[0]:.4f}')
print(f'\tbeta_1: {result.x[1]:.4f} +/- {parameter_error(hess_inv, var_residuals)[1]:.4f}')
print('Covariance matrix of optimal parameters:')
print(covariance(hess_inv, var_residuals))

x_values = np.linspace(min(xs) - 10, max(xs) + 10, 100)

print (result.x[:])
# Use the logistic model to make predictions
predictions = p(x_values, result.x[0], result.x[1])

# Plot the data and the logistic regression model
plt.scatter(xs, ys, label='Data')
plt.plot(x_values, predictions, label='Logistic Regression Model', color='red')
plt.xlabel('Age')
plt.ylabel('Probability of Recognition')
plt.legend()
plt.savefig("p2_3.png")
plt.show()