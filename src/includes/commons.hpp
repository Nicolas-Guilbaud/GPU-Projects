#pragma once

#include <stdio.h>
#include <fstream>
#include <utility>

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


const int DEFAULT_ARRAY_SIZE = 1024,
    DEFAULT_THREAD_SIZE = 1024,
    DEFAULT_J = 1,
    DEFAULT_ITERATIONS = 1,
    DEFAULT_K = 1;

/**
 *  Upper bound division
 */
int div_up(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * Select which metric to use for the benchmark
 */
enum class Metric {
    avg,
    median,
};


class DataPoint {
public:
    float y;
    int x;

    DataPoint() = default;

    DataPoint(float value, int x_axis)
        : y(value), x(x_axis) {
    }

    std::string to_csv() const {
        return std::to_string(y) + "," + std::to_string(x);
    };
};

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

void save_data(std::string filename, const DataPoint* time, int N) {
    std::ofstream fout;
    fout.open(filename);
    if (!fout.good()) {
        printf("Could not open %s !", filename.c_str());
    }

    for (int i = 0; i < N; i++) {
        fout << time[i].to_csv() << "\n";
    }
    fout.close();

}

/**
 * Computes the median value of an array
 */
float compute_median(float* values, size_t nb_iter) {
    int mid = nb_iter / 2;
    if (nb_iter % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    else {
        return values[mid];
    }
}

/**
 * Computes the average value of an array
 */
float compute_avg(float* values, size_t nb_iter) {
    float res = 0;
    for (size_t i = 0; i < nb_iter; i++) {
        res += values[i];
    }
    return res / nb_iter;
}

/**
 * computes the metric base on an array of values.
 */
float compute_metric(Metric choice, float* values, size_t nb_iter) {
    switch (choice) {
    case Metric::avg:
        return compute_avg(values, nb_iter);
    case Metric::median:
        return compute_median(values, nb_iter);
    default:
        return 0.0;
    }
    return 0.0f;
}

