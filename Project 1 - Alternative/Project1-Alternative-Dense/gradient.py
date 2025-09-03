# Gradient Descend Method

### Imports
#### External libraries
import numpy as np
from numpy.linalg import norm
from time import perf_counter

#### Custom libraries
from matrix_utils import check_symmetry, check_pos_def


### Update function
def update_gradient(x : np.ndarray, r : np.ndarray, b : np.ndarray, A : np.ndarray, verbose : bool = False) \
    -> np.ndarray:
    '''
    Update step of the Gradient Descent method.
    '''
    
    y = A.dot(r)
    a = r.T.dot(r)
    b = r.T.dot(y)
    a = a / b
    x = x + a * r

    if verbose:
        print(f"r: {r}")
        print(f"y: {y}")
        print(f"b: {b}")
        print(f"A: {A}")
        print(f"x: {x}")

    return x


def gradient(A : np.ndarray, b : np.ndarray, tol : float, verbose : bool = False) \
    -> tuple[np.ndarray, list, float, float]:
    '''
    Gradient Descent method.
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

    while e > tol:
        x = update_gradient(x, r, b, A)
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