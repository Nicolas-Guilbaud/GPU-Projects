#pragma once

#include <stdio.h>
#include <fstream>

#define CHK(code)                                                    \
    do                                                               \
    {                                                                \
        if ((code) != cudaSuccess)                                   \
        {                                                            \
            fprintf(stderr, "CUDA error: %s %s %i\n",                \
                    cudaGetErrorString((code)), __FILE__, __LINE__); \
            goto Error;                                              \
        }                                                            \
    } while (0)

/**
 *  Upper bound division
 */
int div_up(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * Select which metric to use for the benchmark
 */
enum class metric {
    avg,
    median,
};

/**
 * Select which benchmark to run
 */
enum class benchmark{
    ARRAY_SIZE,         // vary the array size
    MULTIPLE_ELEMS,     // vary the number of elements processed by a single thread
    COMPUTE_INTENSITY   // vary the number of operations per thread
};

/**
 * Select which data type to use
 */
enum class data_type{
    DOUBLE,
    FLOAT
};

/**
 * Saves the data into a csv file.
 * 
 * Params:
 * 
 * - filename: the name of the file
 * - time: the array of time values
 * - N: the size of the time array
 */
void save_data(const char* filename, const float* time, int N) {
    std::ofstream fout;
    fout.open(filename);
    if (!fout.good()) {
        printf("Could not open %s !", filename);
    }

    for (int i = 1; i < N; i++) {
        fout << i << "," << time[i] << "\n";
    }
    fout.close();

}

//FIXME: must be implemented
float compute_median(float* values, size_t nb_iter) {
    return 0;
}

/**
 * Computes the average value of an array
 */
float compute_avg(float* values, size_t nb_iter) {
    float res = 0;
    for (int i = 0; i < nb_iter; i++) {
        res += values[i];
    }
    return res / nb_iter;
}

/**
 * computes the metric base on an array of values.
 */
float compute_metric(metric choice, float* values, size_t nb_iter) {
    switch (choice) {
    case metric::avg:
        return compute_avg(values, nb_iter);
    case metric::median:
        return compute_median(values, nb_iter);
    }
    return 0;
}

//conversion double <-> binary
union bin_double{
    double value;
    u_int64_t binary;
};