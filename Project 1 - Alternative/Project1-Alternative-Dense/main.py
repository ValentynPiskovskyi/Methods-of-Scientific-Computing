import numpy as np
from jacobi import jacobi
from gauss_seidel import gauss_seidel
from gradient import gradient
from gradient_conjugate import gradient_conjugate
from criteria import relative_error


def solve_all_methods(A : np.ndarray, b : np.ndarray, x : np.ndarray, tol : float) :
    '''
    Main function that launches the execution of 4 methods solving linear systems for 
    the same system A*x = b, with the tolerance tol.
    '''

    print("\n--- Jacobi Method ---")
    xh, errors, time, nit = jacobi(A, b, tol)
    rel_error = relative_error(xh, x)
    print("Relative error: ", rel_error)
    print("Number of iterations: ", nit)
    print("Total time (s): ", time)


    print("\n--- Gauss-Seidel Method ---")
    xh, errors, time, nit = gauss_seidel(A, b, tol)
    rel_error = relative_error(xh, x)
    print("Relative error: ", rel_error)
    print("Number of iterations: ", nit)
    print("Total time (s): ", time)


    print("\n--- Gradient Method ---")
    xh, errors, time, nit = gradient(A, b, tol)
    rel_error = relative_error(xh, x)
    print("Relative error: ", rel_error)
    print("Number of iterations: ", nit)
    print("Total time (s): ", time) 


    print("\n--- Gradient Conjugate Method ---")
    xh, errors, time, nit = gradient_conjugate(A, b, tol)
    rel_error = relative_error(xh, x)
    print("Relative error: ", rel_error)
    print("Number of iterations: ", nit)
    print("Total time (s): ", time) 