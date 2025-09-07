# Methods of Scientific Computing 
### Author: Valentyn Piskovskyi (886485)

## Project 1 - Alternative

To execute the any of the programs within this project it is necessary to install the following dependencies. 

- numpy
- scipy

It can be done using the following command.

```bash
    pip install <dependency name>
```

To execute Jacobi, Gaiss-Seidl, gradient and conjugate gradient methods on matrices "spa1.mtx", "spa2.mtx", "vem1.mtx" and "vem2.mtx" with thresholds 1e-4, 1e-6, 1e-8 and 1e-10, it is necessary to launch the "validation.py" script, which can be done in the following way (from the "Project1-Alternative-Sparse" or "Project1-Alternative-Dense" directories).

```bash
    python validation.py
```

The folder "Additional" contains Python notebooks with methods used to analyse matrices and plot the results. The necessary data collected by performing the tests is also included.

## Project 2

To execute the any of the programs within this project it is necessary to install the following dependencies. 

- numpy
- scipy
- pillow
- pyqt5

It can be done using the following command.

```bash
    pip install <dependency name>
```

To launch the methods used to assess the correctnes of the DCT implementation or to measure execution time, it is necessary to launch "test_implementation.py" or "test_time.py" script, respectively. It can be done from the "Part1" directory in the following way. 

```bash
    python test_implementation.py
    python test_time.py
```

The corresponding plots will be save in a "plot" directory, created automatically.

To launch the main application, it is necessary to run the "program.py" script, which can be done in the following way (from the "Part2" directory), which can be done in the following way.

```bash
    python program.py
```

Alternatively, on machines using Windows operating system, it is possible to launch the executable file "program.exe" from the folder "additional/dist".