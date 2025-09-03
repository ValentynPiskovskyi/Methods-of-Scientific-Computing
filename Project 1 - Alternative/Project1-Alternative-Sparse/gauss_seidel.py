# Gauss-Seidl Method

### Imports
#### External libraries
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_array, tril
from scipy.sparse.linalg import spsolve_triangular
from time import perf_counter

#### Custom libraries
from matrix_utils import solve_l, check_symmetry, check_pos_def


def update_gauss_seidel(x : np.ndarray, r : np.ndarray, L : csr_array, verbose : bool = False) -> np.ndarray:
    '''
    Update step of the Gauss-Seidl method.
    '''

    if verbose:
        print(f"x: {x}")
        print(f"r: {r}")
        print(f"L: {L}")

    return solve_l(L, r + L.dot(x)).T # Alternative: return spsolve_triangular(L, L, r + L.dot(x)).T
    

### Method: Gauß-Seidl
def gauss_seidel(A : csr_array, b : np.ndarray, tol : float, verbose : bool = False) \
    -> tuple[csr_array, list, float, float]:
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

    d = A.diagonal()
    if any(abs(d) < 1e-14):
        raise ValueError(f"Error. Zero diagonal entry found in the matrix.")
    
    start = perf_counter()

    max_iter = 21000
    k = 0
 
    x = np.zeros(A.shape[0])
    norm_b = norm(b)
    r = b - A.dot(x)
    e = norm(r) / norm_b
    errors = [e]

    L = tril(A).tocsr()

    while e > tol:
        x = update_gauss_seidel(x, r, L)
        k = k + 1
        r = b - A.dot(x)
        e = norm(r) / norm_b

        if k > max_iter:
            print(f"Error. Maximum number of iterations exceeded. Achieved scaled residual: {e}")
            break

        if verbose:
            print(f"Residual: {e}")
        errors.append(e)
    
    print(f"Total number of iterations: {k}")
    total_seconds = perf_counter() - start

    return x, errors, total_seconds, k     