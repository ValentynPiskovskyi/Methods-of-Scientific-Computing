# Jacobi Method

## Imports
#### External libraries
from scipy.sparse import csr_array
from scipy.sparse.linalg import inv
from time import perf_counter
import numpy as np
from numpy.linalg import norm

#### Custom libraries
from matrix_utils import get_diag, check_symmetry, check_pos_def, inv_diag


## Update
def update_jacobi(x : np.ndarray, r : np.ndarray, D_inv : csr_array, verbose : bool = False) \
    -> np.ndarray:
    '''
    Update step of the Jacobi method.
    '''

    if verbose:
        print(f"x: {x}")
        print(f"r: {r}")
        print(f"D_inv: {D_inv}")
    
    return x + D_inv.dot(r)



## Method: Jacobi for dense matrices
def jacobi(A : csr_array, b : csr_array, tol : float, verbose : bool = False) \
    -> tuple[csr_array, list, float, float]:
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