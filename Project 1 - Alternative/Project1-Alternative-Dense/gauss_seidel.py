# Gauss-Seidl Method

### Imports
#### External libraries
import numpy as np
from numpy.linalg import norm
from time import perf_counter

#### Custom libraries
from matrix_utils import solve_l, check_symmetry, check_pos_def


def update_gauss_seidel(x : np.ndarray, r : np.ndarray, L : np.ndarray, verbose : bool = False) -> np.ndarray:
    '''
    Update step of the Gauss-Seidl method.
    '''

    if verbose:
        print(f"x: {x}")
        print(f"r: {r}")
        print(f"L: {L}")

    return solve_l(L, r + L.dot(x)).T
    

def gauss_seidel(A : np.ndarray, b : np.ndarray, tol : float, verbose : bool = False) \
    -> tuple[np.ndarray, list, float, float]:
    '''
    Gauß-Seidel method.
    '''

    if check_symmetry(A) == False:
        print("Error: The matrix is not symmetric")
        return None
    
    if check_pos_def(A) == False:
        print("Error: The matrix is not positive definite")
        return None
    
    print("The matrix is symmetric and positive definite.")
    
    start = perf_counter()

    max_iter = 21000
    k = 0
 
    x = np.zeros(A.shape[0])
    norm_b = norm(b)
    r = b - A.dot(x)
    e = norm(r) / norm_b
    errors = [e]

    L = np.tril(A)

    while e > tol:
        x = update_gauss_seidel(x, r, L)
        k = k + 1
        r = b - A.dot(x)
        e = norm(r) / norm_b

        if k > max_iter:
            print(f"Error. Maximum number of iterations exceeded. Achieved scaled residual: {e}")
            break

        if verbose:
            print(f"Iteration: {k}")
            print(f"Residual: {e}")

        errors.append(e)
    
    print(f"Total number of iterations: {k}")
    total_seconds = perf_counter() - start

    return x, errors, total_seconds, k   