import numpy as np


''' Threshold use to deal with numeric precision when compari values (including zero comparisons).'''
ZERO_THRESHOLD = 1e-14 


def get_diag(A : np.ndarray) -> np.ndarray:
    '''
    Calculate the diagonal matrix from a given matrix
    '''

    return np.diag(np.diag(A))


def solve_l(U : np.ndarray, b : np.ndarray) -> np.ndarray: 
    '''
    Solve a linear system with the matrix 
    of the lower triangular form
    '''
    
    n = len(b)
    x = np.zeros(n)

    for i in range(n):
        x[i] = (b[i] - np.dot(U[i,:i], x[:i])) / U[i, i]
    
    return x


def inv_diag(D : np.ndarray) -> np.ndarray:
    '''
    Invert a diagonal matrix.
    
    Note: Python automatically manges a division by 0 as an exception. 
    '''

    A = np.zeros((D.shape[0], D.shape[1]))
    np.fill_diagonal(A, 1 / np.diagonal(D)) # Dangerous, automatically managed by Python

    return A


def check_symmetry(A : np.ndarray) -> bool:
    '''
    Check for matrix's symmetry.
    '''

    (A.T - ZERO_THRESHOLD <= A).all() and (A <= A.T + ZERO_THRESHOLD).all()


def check_pos_def(A : np.ndarray) -> bool:
    '''
    Check for matrix's positive definiteness using Cholesky decomposition.
    A positive response is returned if the Cholesky's decomposition is correctly 
    performed. In case exceptions are raised, a negative response is returned.
    '''

    try: 
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False