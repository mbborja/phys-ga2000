import numpy as np
import timeit

def madelung_slow(L):
    """This solves for the madelung constant in an L x L x L space
    
    L (Integer): Dimensions of L x L x L space

    Returns:
        M: Final Madelung constant
    """
    M = 0
    
    # For loop over all space
    for i in range (-L , L + 1):
        for j in range (-L , L + 1):
            for k in range (-L, L + 1):
                
                if i == 0 and j == 0 and k == 0: # Origin condition
                    continue
                else:    
                    if (i + j + k) % 2 == 0: # Distance calculation is positive if i + j + k is even
                        M += np.sqrt(i**2 + j**2 + k**2)
                    else: # Negative if it is odd
                        M -= np.sqrt(i**2 + j**2 + k**2)
                    
    return M

def madelung_fast(L):
    """This madelung constant solver uses numpy arrays and meshgrids to speed up calculations

    Args:
        L (Integer): Dimensions of L x L x L space

    Returns:
        M: Final Madelung constant
    """
    
    x, y, z = np.meshgrid(np.arange(-L, L + 1), np.arange(-L, L + 1), np.arange(-L, L + 1))
    distances = np.sqrt(x**2 + y**2 + z**2)
    even_mask = (x + y + z) % 2 == 0
    odd_mask = ~even_mask
    
    # Exclude the origin (center)
    origin_index = (L, L, L)
    distances[origin_index] = 1  # Set the value at the origin to 1
    
    distances[even_mask] = 1 / distances[even_mask]
    distances[odd_mask] = -1 / distances[odd_mask]
    
    distances[(L,L,L)] = 0 # Set the value at the origin to 0 since it shouldn't contribute
    
    M = np.sum(distances)
    return M