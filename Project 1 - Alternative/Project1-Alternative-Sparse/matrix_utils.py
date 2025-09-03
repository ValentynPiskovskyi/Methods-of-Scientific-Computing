from scipy.sparse import csr_array, diags
import numpy as np
from scipy.sparse.linalg import eigsh
import numpy as np


''' Threshold use to deal with numeric precision when compari values (including zero comparisons).'''
ZERO_THRESHOLD = 1e-14 


def get_diag(A : csr_array) -> csr_array:
    '''
    Calculate the diagonal matrix from a given matrix
    '''

    return diags(A.diagonal()).tocsr()


def solve_l_dot(L, b: np.ndarray) -> np.ndarray:
    '''
    Solve a linear system with the matrix of the lower triangular form, 
    using dot products between matrix's rows and the x vector.
    '''

    n = L.shape[0]
    x = np.zeros(n)

    for i in range(n):
        row_start = L.indptr[i]
        row_end = L.indptr[i + 1]
        row_indices = L.indices[row_start:row_end]
        row_data = L.data[row_start:row_end]

        mask = row_indices < i
        if mask.any():
            sum_ = np.dot(row_data[mask], x[row_indices[mask]])
        else:
            sum_ = 0.0

        diag_mask = row_indices == i
        if not np.any(diag_mask):
            raise ValueError(f"Method solve_l: Zero diagonal entry found in the matrix at entry {i}.")
        diag = row_data[diag_mask][0]

        x[i] = (b[i] - sum_) / diag

    return x


def solve_l(U : csr_array, b : np.ndarray) -> np.ndarray:
    '''
    Solve a linear system with the matrix of the lower triangular form, 
    using explicit iteration over matrix's row elements.
    '''

    n = U.shape[0]
    x = np.zeros(n)

    U = U.tocsr()

    for i in range(n):
        row_start = U.indptr[i]
        row_end = U.indptr[i + 1]
        indices = U.indices[row_start:row_end]
        data = U.data[row_start:row_end]
        
        sum = 0
        diagonal = data[-1] # Diagonal element is always the last one on the row.
        
        # Note: j > i cannot occur given that the matrix is lower triangular
        for k, j in enumerate(indices):
            val = data[k]
            if j < i:
                sum += val * x[j]
        
        x[i] = (b[i] - sum) / diagonal

    return x


def inv_diag(D : csr_array) -> csr_array:
    '''
    Invert a diagonal matrix.
    
    Note: Python automatically manges a division by 0 as an exception. 
    '''
    
    diag = D.diagonal()
    
    if np.any(diag == 0):
        raise ValueError("Error. Matrix has zeros on the diagonal.")
    
    inv_diag = 1.0 / diag # Dangerous, automatically managed by Python
    return diags(inv_diag, format="csr")


def check_symmetry(A : csr_array) -> bool:
    '''
    Check for matrix's symmetry.
    '''
    
    return (abs(A - A.T)).nnz <= ZERO_THRESHOLD


def check_pos_def(A : csr_array) -> bool:
    '''
    Check for matrix's positive definiteness by computing the smallest egienvalue.
    An exception is raised in case the matrix is not positive definite (i.e. the 
    decomposition cannot be performed).

    Note: The implementation of Cholesky decomposition with sparse matrices is not trivial, 
    due to the necessity of managing the fill-in problem.
    '''

    try:
        λ_min = eigsh(A, k=1, which='SA', return_eigenvectors=False)
        return λ_min[0] > ZERO_THRESHOLD
    except Exception as e:
        print("SPD check exception: ", e)
        return False