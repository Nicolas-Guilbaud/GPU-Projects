CUDA Mini Project
=================

This project consists of analysing the performances of the XOR bitwise operation on an array of `float` and `doubles` data types on a NVIDIA GPU. It measures the performances (computational and memory) by varying either:

- the size of the arrays
- the number of elements processed per thread
- the number of operations (workload) per thread

# 1. Prerequisites

The following are required to build the project:

- a CUDA capable GPU
- CMake v.3.20 or greater

# 2. Compiling the binary

In order to compile the source code, you can just run `cmake --build ./build` from the current folder.  

If you use vscode with the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) extension, just run `CMake: clean rebuild` from the command palette.

# 2. Usage

All the benchmarks executions are bundled in one simple binary via CMake. Hence, arguments must be passed to select which test to perform on which data type. Theses are parsed via the [CLI11](https://github.com/CLIUtils/CLI11) library, whose header is provided in the `extern` folder.

A summary of the arguments are found in the following table:

| Argument     | Abbreviated | Description                                               | Default value |
|--------------|-------------|-----------------------------------------------------------|---------------|
| --help       | -h          | Prints the help message                                   |               |
| --thread     | -t          | Maximal number of threads per block                       | 1024          |
| --metric     | -m          | Metric to use for performance measurement (avg or median) | avg           |
| --max_size   | -n          | Upper bound on the array size                             | 1             |
| --iterations | -i          | Number of times to probe the measurements                 | 1             |
| --J          | -j          | Number of elements per threads to process                 | 1             |
| --K          | -k          | Number of operations per threads to process               | 1             |
| --output     | -o          | Generic name for the csv output files                     | "output"      |
| --steps      | -s          | The steps for the array size                              | 1             |
| --step_j     |             | The steps for the number of elements per threads          | 1             |
| --step_k     |             | The steps for the number of operations per threads        | 1             |
| --double     | -d          | Run the benchmarks with the double data type              |               |
| --float      | -f          | Run the benchmarks with the float data type               |               |

Note: if not enough argument is specified, the program will just exit without doing anything.

The results, which are obtained in a `csv` format in the `result directory`, can be automatically plotted on a graph via the `plotter.py` python file. Theses are then exported in a `png` file of the same name in the same directory.