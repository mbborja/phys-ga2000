import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brent

f = lambda x: (x - 0.3)**2 * np.exp(x)

def golden_mean_search(a, b):
    golden_ratio = (1 + np.sqrt(5)) / 2
    epsilon = 1e-7
    tol = 1e-7

    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

    return (b + a) / 2

def optimize():
    a = -1
    b = 2
    tol = 1e-7

    if abs(f(a)) < abs(f(b)):
        a, b = b, a

    c = a
    flag = True
    err = abs(b - a)
    err_list, b_list = [err], [b]

    while err > tol:
        s = golden_mean_search(a, b)

        if ((s >= b)) or ((flag == True) and (abs(s - b) >= abs(b - c))) or (
                (flag == False) and (abs(s - b) >= abs(c - d))):
            s = (a + b) / 2
            flag = True
        else:
            flag = False

        c, d = b, c

        a = s

        if abs(f(a)) < abs(f(b)):
            a, b = b, a

        err = abs(b - a)
        err_list.append(err)
        b_list.append(b)

    print(f'minimum = {b}')
    return b_list, err_list

def plot(b_list, err_list):
    log_err = [np.log10(err) for err in err_list]
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axs[0], axs[1]

    ax0.scatter(range(len(b_list)), b_list, marker='o', facecolor='red', edgecolor='k')
    ax0.plot(range(len(b_list)), b_list, 'r-', alpha=.5)
    ax1.plot(range(len(err_list)), log_err, '.-')
    ax1.set_xlabel('number of iterations')
    ax0.set_ylabel(r'$x_{min}$')
    ax1.set_ylabel(r'$\log{\delta}$')
    plt.savefig('convergence.png')

if __name__ == "__main__":
    b_list, err_list = optimize()
    plot(b_list, err_list)

    # Compare with scipy.optimize.brent
    result_scipy = brent(f)
    print(f'Scipy result: {result_scipy}')