# Gradient Conjugate Method

### Imports
#### External libraries
from scipy.sparse import csr_array 
from time import perf_counter
import numpy as np
from numpy.linalg import norm

#### Custom libraries
from matrix_utils import check_symmetry, check_pos_def


def update_gradient_conjugate(x : np.ndarray, r : np.ndarray, d : np.ndarray, 
                              b : np.ndarray, A : csr_array, verbose : bool = False) \
    -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Update step of the Conjugate Gradient method.
    '''

    y = A.dot(d)
    alpha = d.T.dot(r) / d.T.dot(y)
    x = x + alpha * d
    r = b - A.dot(x)
    w = A.dot(r)
    beta = d.T.dot(w) / d.T.dot(y)
    d = r - beta * d
    if verbose:
        print(f"alpha: {alpha}")
        print(f"x: {x}")
        print(f"r: {r}")
        print(f"beta: {beta}")
        print(f"d: {d}")

    return x, d, r


def gradient_conjugate(A : csr_array, b : np.ndarray, tol : float, verbose : bool = False) \
    -> tuple[csr_array, list, float, float]:
    '''
    Conjugate Gradient method.
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

    d = r

    while e > tol:
        x, d, r = update_gradient_conjugate(x, r, d, b, A)
        k = k + 1
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