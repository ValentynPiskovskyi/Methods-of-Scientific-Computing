import numpy
import os
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from PIL import Image
from dct import dct, idct, dct2, idct2, dct2_alt, idct2_alt, dct2_naive, idct2_naive
from dct import dct2


def test_dct():
    '''
    Test various implementations of the one-dimensional DCT and IDCT 
    applying them to a pre-defined array.
    '''

    print("\n---- Test of the DCT and IDCT with custom and library-derived implementations ----")
    d = numpy.array([231, 32, 233, 161, 24, 71, 140, 245], dtype=numpy.float_)
    dct_res_custom = dct(d)
    dct_res_lib = dctn(d, norm="ortho")
    print("Input array: \n", d)
    print("Custom DCT: \n", dct_res_custom)
    print("Library DCT: \n", dct_res_lib)
    print("Custom IDCT: \n", idct(dct_res_custom))
    print("Library IDCT: \n", idctn(dct_res_lib, norm = "ortho"))


def test_dct2():
    '''
    Test various implementations of the DCT2 and IDCT2 
    applying them to a pre-defined matrix.
    '''

    print("\n---- Test of the DCT2 and IDCT2 with custom and library-derived implementations ----")
    D = numpy.array([[231, 32, 233, 161, 24, 71, 140, 245],
                [247, 40, 248, 245, 124, 204, 36, 107], 
                [234, 202, 245, 167, 9, 217, 239, 173], 
                [193, 190, 100, 167, 43, 180, 8, 70],
                [11, 24, 210, 177, 81, 243, 8, 112], 
                [97, 195, 203, 47, 125, 114, 165, 181], 
                [193, 70, 174, 167, 41, 30, 127, 245],
                [87, 149, 57, 192, 65, 129, 178, 228]], dtype=numpy.float64)
    dct2_res_custom = dct2(D)
    dct2_res_custom_alt = dct2_alt(D)
    dct2_res_custom_naive = dct2_naive(D)
    dct2_res_lib = dctn(D, norm="ortho")
    print("Input matrix: \n", D)
    print("Custom DCT2: \n", dct2_res_custom)
    print("Custom DCT2 Alt: \n", dct2_res_custom_alt)
    print("Custom DCT2 Naive: \n", dct2_res_custom_naive)
    print("Library DCT2: \n", dct2_res_lib)
    print("Custom IDCT2: \n", idct2(dct2_res_custom))
    print("Custom IDCT2 Alt: \n", idct2_alt(dct2_res_custom_alt))
    print("Custom IDCT2 Naive: \n", idct2_naive(dct2_res_custom_naive))
    print("Library IDCT2: \n", idctn(dct2_res_lib, norm = "ortho"))


def test_image_dct2():
    '''
    Test various implementations of the DCT2 and IDCT2 
    applying them to an 8 x 8 block extracted from an image.
    '''
    
    print("\n---- Test of the DCT2 and IDCT2 with custom and library-derived implementations ----")

    # Load an image annd extract an 8 x 8 (upper left) block as an input matrix
    img = numpy.copy(Image.open("images/surfer.png").convert("L"))
    A = numpy.array(img[:8, :8], numpy.float64).copy()

    res_custom = dct2(A)
    res_lib = dctn(A, norm = "ortho")
    
    print("8 x 8 portion of the input matrix A \n", A)
    print("Custom DCT2 on image: \n", res_custom)
    print("Library DCT2 on image: \n", res_lib)
    print("Custom IDCT2 on image: \n", idct2(res_custom))
    print("Library IDCT2 on image: \n", idctn(res_lib, norm = "ortho"))
    print("Mean difference: ", numpy.mean(abs(res_custom - res_lib)))

    # Create plots of coefficients produced by the custom DCT2, library-based version
    # and the difference between the two resulting matrices
    plot3d(res_custom, "dct_coeff_custom.png")
    plot3d(res_lib, "dct_coeff_lib.png")
    plot3d(res_custom - res_lib, "dct_coeff_diff.png")


def plot3d(matrix, filename = "output.png"):
    '''
    Create a 3D plot of a matrix specified as an input.
    '''

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection = "3d")

    x, y = numpy.meshgrid(numpy.arange(matrix.shape[1]), 
                    numpy.arange(matrix.shape[0]))
    x = x.ravel()
    y = y.ravel()
    z = numpy.zeros_like(x)

    ax.bar3d(x, y, z, 0.8, 0.8, matrix.ravel())
    os.makedirs("plots", exist_ok = True)
    plt.savefig("plots/" + filename)


if __name__ == "__main__":
    test_dct()
    test_dct2()
    test_image_dct2()