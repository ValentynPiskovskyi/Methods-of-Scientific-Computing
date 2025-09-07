### Imports
#### External libraries
import os
import argparse
import numpy as np
from scipy.io import mmread

#### Custom libraries
from main import solve_all_methods



def validate(data_path : str = "data"):
    '''
    Validation function that load 4 matrices from the "data" subfolder. Launches 4 methods 
    solving linear systems on the same matrix A, r.h.s. vector b of all ones and for 
    10^-4, 10^-6, 10^-8, 10^-10 tolarance values.
    Produces as an output 
    - elapsed time, 
    - number of iterations,
    - relative error.
    '''

    for mat_name in os.listdir(data_path):
        print(f"\n#### Matrix: {mat_name} ####")
        
        # Reads the matrix
        A = np.array(mmread(f"{data_path}\\{mat_name}").todense())

        if A.shape[0] >= 3 and A.shape[1] >= 3:
            print("A: ", A[:3, :3])

        sparcity = np.count_nonzero(np.abs(A.data) > 1e-14) / (A.shape[0] * A.shape[1])
        print(f"Sparcity of the matrix A: {sparcity}")

        # Exact solution
        x = np.ones((A.shape[0]))
        print("x: ", x.shape)

        # Right-hand side for the exact solution
        b = np.dot(A, x)
        print("b shape: ", b.shape)
        print("b: ", b[:10])

        for tol in [10**-4, 10**-6, 10**-8, 10**-10] :
            print(f"\n ### Tolerance: {tol} ###")
            solve_all_methods(A, b, x, tol)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Linear System Solver',
                    description='Library for solving linear systems')
    
    parser.add_argument("--data_path", default = "data", help = "Path to the directory contatining coefficient matrices.")
    args = parser.parse_args()
    print("ARGS: ", args.data_path)

    validate(args.data_path)