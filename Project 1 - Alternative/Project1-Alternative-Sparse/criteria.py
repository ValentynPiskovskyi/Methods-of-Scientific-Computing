from scipy.sparse import csr_array
from numpy.linalg import norm
import numpy as np

def scaled_residual(A : csr_array, b : np.ndarray, x : np.ndarray, verbose : bool = False) -> float:
    '''
    Calculate the residual between the exact and 
    approximated solutions, scaled by the r.h.s. vector b.
    '''

    if verbose:
        print(f"||b - Ax|| = {norm(b - A.dot(x))}")
    
    return norm(A.dot(x) - b) / norm(b) 


def relative_error(xh : np.ndarray, x : np.ndarray, verbose : bool = False) -> np.ndarray:
    '''
    Calculate the relative error between the exact 
    and approximated solution, scaled by the exact solution.
    '''
    
    if verbose:
        print(f"||xh - x|| = {norm(xh - x) / norm(x)}")
    
    return norm(xh - x) / norm(x)