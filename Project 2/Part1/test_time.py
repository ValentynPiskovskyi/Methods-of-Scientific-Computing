import numpy
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.fft import dctn, idctn
from dct import dct2, idct2, dct2_alt, dct2_naive, idct2_alt


TYPE = "basic_experiment"


def make_test_matrix(N):
    rng = numpy.random.default_rng(42)
    return numpy.array(rng.normal(size = (N, N)), dtype=numpy.float64)


def make_test_matrix_uniform(N):
    rng = numpy.random.default_rng(42)
    return rng.uniform(0, 255, size=(N, N))


def make_constant_matrix(N):
    return numpy.full((N, N), 1.0)


def make_color_gradient_matrix(N):
    x = numpy.linspace(0, 255, N)
    return numpy.tile(x, (N, 1))


def make_checkered_matrix(N):
    A = numpy.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            A[i, j] = (i + j) % 2
    
    return A


def run_test():
    '''
    Run various DCT2 implementations on matrices of increasing sizes, 
    defined according to a specific strategy.
    '''

    time_custom = []
    time_custom_alternative = []
    time_custom_naive = []
    time_lib = []

    dimensions = range(100, 1050, 50) # (100, 1700, 50)
    #dimensions = [2**x for x in range(3, 15)] # Array of powers of 2 as dimensions

    for n in tqdm(dimensions):
        A = make_test_matrix(n)

        # Custom implementation
        start = time.perf_counter()
        dct2(A)
        sec = time.perf_counter()
        time_custom.append(sec - start)

        # Custom (Alternative) implementation
        start = time.perf_counter()
        #dct2_alt(A)
        sec = time.perf_counter()
        time_custom_alternative.append(sec - start)

        # Custom (Naive) implementation
        start = time.perf_counter()
        #dct2_naive(A)
        sec = time.perf_counter()
        time_custom_naive.append(sec - start)

        # Implementation by SciPy
        start = time.perf_counter()
        dctn(A, norm = "ortho")
        sec = time.perf_counter()
        time_lib.append(sec - start)

    return dimensions, time_custom, time_custom_alternative, time_custom_naive, time_lib 


def create_plots(dimensions, time_lib, time_custom, time_custom_alt):
    '''
    Create plots of measured and theoretic time curves for 
    the library-based and custom DCT2 implementations.
    '''

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_yscale("log")
    colors = plt.get_cmap("tab10").colors
    
    # Plot recorded times
    ax.plot(dimensions, time_lib, label="SciPy implementation", color=colors[1], marker='o')
    ax.plot(dimensions, time_custom, label="Custom implementation", color=colors[0], marker='s')

    # Prepare and normalize theoretical complexity curves
    sizes_arr = numpy.array(dimensions, dtype=numpy.float64)
    scale_index = int(len(sizes_arr) / 2)
    t1 = sizes_arr**2 * numpy.log2(sizes_arr)
    t2 = sizes_arr**3
    t1 = t1 / t1[scale_index] * time_lib[scale_index]
    t2 = t2 / t2[scale_index] * time_custom[scale_index]

    # Plot theoretical curves with dashed lines
    ax.plot(dimensions, t1, label=r"$\mathcal{O}(N^2 \log N)$ scaled", color=colors[1], linestyle='--')
    ax.plot(dimensions, t2, label=r"$\mathcal{O}(N^3)$ scaled", color=colors[0], linestyle='--')

    # Labels and formatting
    ax.set_xlabel("Matrix size (N)", fontsize=14)
    ax.set_ylabel("Time (s, log scale)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Legend formatting
    ax.legend(fontsize=12, ncol=2)

    # Save with safe and unique filename
    os.makedirs(f"runs_{TYPE}", exist_ok=True)
    plot_id = len([f for f in os.listdir(f"runs_{TYPE}") if f.startswith("time") and f.endswith(".png")])
    plt.tight_layout()
    plt.savefig(f"runs_{TYPE}/time_mat{plot_id}.png", dpi=150)
    #plt.show()
    plt.close()



def create_plots_custom(dimensions, time_custom, time_custom_alternative, time_custom_naive):
    '''
    Create plots of measured and theoretic time curves for 
    various custom DCT2 variations.
    '''

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_yscale("log")
    colors = plt.get_cmap("tab10").colors

    # Plot recorded times
    ax.plot(dimensions, time_custom, label="Custom", color=colors[0], marker='^')
    ax.plot(dimensions, time_custom_alternative, label="Custom (Alternative)", color=colors[2], marker='^')
    ax.plot(dimensions, time_custom_naive, label="Custom (Naive)", color=colors[4], marker='^')

    # Prepare and normalize theoretical complexity curves
    sizes_arr = numpy.array(dimensions, dtype=numpy.float64)
    scale_index = int(len(sizes_arr) / 2)
    t = sizes_arr**3
    t1 = t / t[scale_index] * time_custom[scale_index]
    t2 = t / t[scale_index] * time_custom_alternative[scale_index]
    t3 = t / t[scale_index] * time_custom_naive[scale_index]

    # Plot theoretical curves with dashed lines
    ax.plot(dimensions, t1, label=r"$\mathcal{O}(N^3)$ scaled to custom", color=colors[0], linestyle='--', alpha = 0.7)
    ax.plot(dimensions, t2, label=r"$\mathcal{O}(N^3)$ scaled to custom (alt.)", color=colors[2], linestyle='--', alpha = 0.7)
    ax.plot(dimensions, t3, label=r"$\mathcal{O}(N^3)$ scaled to custom (naive)", color=colors[4], linestyle='--', alpha = 0.7)

    # Labels and formatting
    ax.set_xlabel("Matrix size (N)", fontsize=14)
    ax.set_ylabel("Time (s, log scale)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Legend formatting
    ax.legend(fontsize=12, ncol=2)

    # Save with safe and unique filename
    os.makedirs(f"runs_{TYPE}", exist_ok=True)
    plot_id = len([f for f in os.listdir(f"runs_{TYPE}") if f.startswith("time") and f.endswith(".png")])
    plt.tight_layout()
    plt.savefig(f"runs_{TYPE}/time_mat_custom{plot_id}.png", dpi=150)
    #plt.show()
    plt.close()


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
    import pickle

    LOAD_FROM_FILE = False

    if LOAD_FROM_FILE:
        with open("runs/dimensions.pkl", "rb") as f:
            dimensions = pickle.load(f)

        with open("runs/time_custom.pkl", "rb") as f:
            time_custom = pickle.load(f)

        with open("runs/time_custom_alternative.pkl", "rb") as f:
            time_custom_alternative = pickle.load(f)

        with open("runs/time_custom_naive.pkl", "rb") as f:
            time_custom_naive = pickle.load(f)

        with open("runs/time_lib.pkl", "rb") as f:
            time_lib = pickle.load(f)
    else:
        dimensions, time_custom, time_custom_alternative, time_custom_naive, time_lib  = run_test()


    create_plots(dimensions, time_lib, time_custom, time_custom_alternative)
    #create_plots_custom(dimensions, time_custom, time_custom_alternative, time_custom_naive)


    import pickle

    with open(f"runs_{TYPE}/dimensions.pkl", "wb") as f:
        pickle.dump(dimensions, f)

    with open(f"runs_{TYPE}/time_custom.pkl", "wb") as f:
        pickle.dump(time_custom, f)

    with open(f"runs_{TYPE}/time_custom_alternative.pkl", "wb") as f:
        pickle.dump(time_custom_alternative, f)

    with open(f"runs_{TYPE}/time_custom_naive.pkl", "wb") as f:
        pickle.dump(time_custom_naive, f)

    with open(f"runs_{TYPE}/time_lib.pkl", "wb") as f:
        pickle.dump(time_lib, f)

