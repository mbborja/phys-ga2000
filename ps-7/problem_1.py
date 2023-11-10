import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brent, minimize_scalar

# Given function y = (x - 0.3)^2 * exp(x)
f = lambda x: (x - 0.3) ** 2 * np.exp(x)

def golden_section_search(a, b, tol=1e-7):
    GOLDEN_RATIO = (np.sqrt(5) + 1) / 2.0

    c = b - (b - a) / GOLDEN_RATIO
    d = a + (b - a) / GOLDEN_RATIO

    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / GOLDEN_RATIO
        d = a + (b - a) / GOLDEN_RATIO

    return (a + b) / 2

def brents_method(f, a, b, tol=1e-7, max_iter=1000):
    c = b - (b - a) / 2.0
    d = c
    e = 0.0

    # Main loop
    for i in range(max_iter):
        tol1 = tol * (abs(c) + 1.0)
        tol2 = 2.0 * tol1

        # Check for convergence
        if abs(c - (a + b) / 2.0) < (tol2 - 0.5 * (b - a)):
            break

        if abs(e) > tol1:
            # Construct a parabolic fit
            r = (b - a) * (f(a) - f(b))
            q = (b - c) * (f(a) - f(c))
            p = (c - a) * (f(b) - f(c))
            s = q - r
            t = q - p
            u = r - p
            if q > 0:
                s = -s
                t = -t
                u = -u

            if 2.0 * s < 3.0 * t and s < t and u < 2.0 * s:
                d = golden_section_search(b, c)
                e = d - c
            else:
                if s < (t - u) / 2.0:
                    d = golden_section_search(b, c)
                    e = d - c
                else:
                    d = (c + e) / 2.0
                    e = d - c
        else:
            d = golden_section_search(b, c)
            e = d - c

        # Update a, b, c
        if abs(e) < tol1:
            if d >= (a + b) / 2.0:
                d = golden_section_search(b, c)
                e = d - c
            else:
                d = golden_section_search(a, c)
                e = d - c

        if abs(d - c) < tol1:
            d = (a + b) / 2.0
            e = 0.0

        a = b
        b = c
        c = d

    return c

def plot(b_list, err_list):
    log_err = [np.log10(err) for err in err_list]
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axs[0], axs[1]
    # plot root
    ax0.scatter(range(len(b_list)), b_list, marker='o', facecolor='red', edgecolor='k')
    ax0.plot(range(len(b_list)), b_list, 'r-', alpha=.5)
    ax1.plot(range(len(err_list)), log_err, '.-')
    ax1.set_xlabel('number of iterations')
    ax0.set_ylabel(r'$x_{min}$')
    ax1.set_ylabel(r'$\log{\delta}$')
    plt.show()
    plt.savefig('convergence.png')

if __name__ == "__main__":
    # Test the modified Brent's method with Golden Mean Search
    a = -2
    b = 2

    # Test the modified Brent's method
    x_min_brent, f_min_brent, _, _ = brent(f, brack=(a, b), full_output=True)

    # Test scipy.optimize.brent
    result_scipy = minimize_scalar(f, bracket=(a, b), method='brent')

    # Print results
    print("Modified Brent's method result:")
    print("Minimum x:", x_min_brent)
    print("Minimum f(x):", f_min_brent)

    print("\nscipy.optimize.brent result:")
    print(result_scipy)

def optimize():
    # Define interval
    a = -10
    b = 10
    tol = 1e-7
    if abs(f(a)) < abs(f(b)):
        a, b = b, a  # Swap bounds
    c = a
    flag = True
    err = abs(b - a)
    err_list, b_list = [err], [b]
    while err > tol:
        s = golden_section_search(a, b)
        if ((s >= b)) or ((flag == True) and (abs(s - b) >= abs(b - c))) or (
                (flag == False) and (abs(s - b) >= abs(c - d))):
            s = (a + b) / 2  # Bisection
            flag = True
        else:
            flag = False
        c, d = b, c  # d is c from the previous step
        a = s
        if abs(f(a)) < abs(f(b)):
            a, b = b, a  # Swap if needed
        err = abs(b - a)  # Update error to check for convergence
        err_list.append(err)
        b_list.append(b)
    print(f'minimum = {b}')
    return b_list, err_list


    # Plot results for the modified Brent's method
    b_list, err_list = optimize()
    plot(b_list, err_list)