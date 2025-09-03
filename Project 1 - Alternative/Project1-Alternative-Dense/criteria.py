import numpy as np
from numpy.linalg import norm

def scaled_residual(A : np.ndarray, b : np.ndarray, x : np.ndarray, verbose : bool = False) -> float:
    '''
    Calculate the residual between the exact and approximated 
    solution, scaled by the r.h.s. vector b.
    '''

    if verbose:
        print(f"||b - Ax|| = {norm(b - np.dot(A, x))}")

    
    return norm(np.dot(A, x) - b) / norm(b) 



def relative_error(xh : np.ndarray, x : np.ndarray, verbose : bool = False) -> float:
    '''
    Calculate the relative error between the exact 
    and approximated solution, scaled by the exact solution.
    '''

    if verbose:
        print(f"||xh - x|| = {norm(xh - x) / norm(x)}")
    
    return norm(xh - x) / norm(x)