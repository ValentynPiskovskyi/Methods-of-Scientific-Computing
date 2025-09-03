# Jacobi Method

## Imports
#### External libraries
import numpy as np
from numpy.linalg import norm, inv
from time import perf_counter

#### Custom libraries
from matrix_utils import get_diag, check_symmetry, check_pos_def, inv_diag


def update_jacobi(x : np.ndarray, r : np.ndarray, D_inv : np.ndarray, verbose : bool = False) \
    -> np.ndarray:
    '''
    Update step of the Jacobi method.
    '''

    if verbose:
        print(f"x: {x}")
        print(f"r: {r}")
        print(f"D_inv: {D_inv}")

    return x + D_inv.dot(r)


def jacobi(A : np.ndarray, b : np.ndarray, tol : float, verbose : bool = False) \
    -> tuple[np.ndarray, list, float, float]:
    '''
    Jacobi method.
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
    
    D = get_diag(A)
    D_inv = inv(D)

    while e >= tol:
        x = update_jacobi(x, r, D_inv)
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