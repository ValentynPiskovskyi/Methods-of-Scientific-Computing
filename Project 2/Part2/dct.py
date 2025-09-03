import numpy

### Main DCT, IDCT, DCT2, IDCT2 methods

def compute_D(n : numpy.float64):
    '''
    Compute a supporting matrix D, used for a matrix-based 
    calculation of DCT, IDCT, DCT2 and IDCT2.
    '''

    if n <= 0:
        raise ValueError("Error! Wrong matrix dimension.")
    
    alpha = numpy.zeros(n, dtype = numpy.float64)
    alpha[0] = 1 / numpy.sqrt(n)
    alpha[1:] = (numpy.sqrt(2) / numpy.sqrt(n))

    D = numpy.zeros((n, n), dtype = numpy.float64)

    for k in range(n):
        for i in range(n):
            D[k, i] = alpha[k] * \
                      numpy.cos(k * numpy.pi * 
                            (2 * i + 1) / (2 * n))
    return D


def dct(f : numpy.ndarray, 
        D : numpy.ndarray = None) -> numpy.ndarray :
    '''
    Compute one-dimensional DCT in matrix form.
    '''


    if f.ndim != 1:
        raise ValueError("Wrong input dimension for a one-dimensional DCT.")
    if D is None:
        n = f.shape[0]
        D = compute_D(n)
    c = numpy.dot(D, f)
    return c


def idct(c : numpy.ndarray, 
         D : numpy.ndarray = None) -> numpy.ndarray :
    '''
    Compute one-dimensional IDCT in matrix form.
    '''
    if c.ndim != 1:
        raise ValueError("Wrong input dimension for a one-dimensional IDCT.")
    if D is None:
        n = c.shape[0]
        D = compute_D(n)
    f = numpy.dot(D.T, c)
    return f


def dct2(F : numpy.ndarray) -> numpy.ndarray:
    '''
    Compute (two-dimensional) DCT2 in matrix form.
    '''

    n, m = F.shape
    D_n = compute_D(n)

    if n == m:
        D_m = D_n
    else: 
        D_m = compute_D(m)

    return numpy.dot(numpy.dot(D_n, F), D_m.T)


def idct2(C : numpy.ndarray) -> numpy.ndarray:
    '''
    Compute (two-dimensional) IDCT2 in matrix form.
    '''

    n, m = C.shape
    D_n = compute_D(n)

    if n == m:
        D_m = D_n
    else: 
        D_m = compute_D(m)
    
    return numpy.dot(numpy.dot(D_n.T, C), D_m)


### Additional DCT2, IDCT2 variants


def dct2_alt(F : numpy.ndarray) -> numpy.ndarray:
    '''
    Compute DCT2 by performing calls to the custom implementation of DCT.
    '''

    n, m = F.shape
    D_n = compute_D(n)
    if n == m:
        D_m = D_n
    else: 
        D_m = compute_D(m)
        
    C = F.copy()
        
    for j in range(m):
        C[:, j] = dct(C[:, j], D_n)
    for i in range(n):
        C[i, :] = dct(C[i, :], D_m)
    
    return C


def idct2_alt(C : numpy.ndarray) -> numpy.ndarray:
    '''
    Compute IDCT2 by performing calls to the custom implementation of IDCT.
    '''
    
    n, m = C.shape
    D_n = compute_D(n)
    if n == m:
        D_m = D_n
    else: 
        D_m = compute_D(m)
    
    F = C.copy()

    for j in range(m):
        F[:, j] = idct(F[:, j], D_n)
    for i in range(n):
        F[i, :] = idct(F[i, :], D_m)
    
    return F


def dot(a, b):
    '''
    Custom (unoptimized) implementation of the dot product of vectors.
    '''

    res = 0
    v1 = a.flatten()
    v2 = b.flatten()

    for i in range(len(v1)):
        res = res + v1[i] * v2[i]
    
    return res


def dct2_naive(F : numpy.ndarray) -> numpy.ndarray:
    '''
    Custom (naive) implementation of the DCT2 relying on the unoptimized implementation 
    of the dot product of vectors.
    '''

    n, m = F.shape
    D_n = compute_D(n)
    if n == m:
        D_m = D_n
    else: 
        D_m = compute_D(m)
        
    C = F.copy()
    C_tmp = F.copy()
        
    for j in range(m):
        for i in range(n):
            C_tmp[i, j] = dot(D_n[i, :], F[:, j])
    for i in range(n):
        for j in range(m):
            C[i, j] = dot(C_tmp[i, :], D_m[j, :])
    
    return C


def idct2_naive(C : numpy.ndarray) -> numpy.ndarray:
    '''
    Custom (naive) implementation of the IDCT2 relying on the unoptimized implementation 
    of the dot product of vectors.
    '''

    n, m = C.shape
    D_n = compute_D(n)
    if n == m:
        D_m = D_n
    else: 
        D_m = compute_D(m)
        
    F = C.copy()
    F_tmp = C.copy()
        
    for j in range(m):
        for i in range(n):
            F_tmp[i, j] = dot(D_n[:, i], C[:, j])
    for i in range(n):
        for j in range(m):
            F[i, j] = dot(F_tmp[i, :], D_m[:, j])
    
    return F