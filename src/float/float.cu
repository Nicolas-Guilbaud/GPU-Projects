#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "extern/CLI11.hpp"
#include "includes/commons.hpp"

using std::rand;

int DEFAULT_ARRAY_SIZE = 1024;
int DEFAULT_THREAD_SIZE = 1024;
int DEFAULT_J = 1;
int DEFAULT_ITERATIONS = 1;
int DEFAULT_K = 1;

union bin_float {
    float f;
    u_int32_t u;
};


bool checkCuda(int* out_cpu, int* out_gpu, int N);

__global__ void float_bitwiseXOR_kernel_k(float* c, const float* a, const float* b, const int N, const int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < K; k++) {
        if (idx < N) {
            bin_float bin_a, bin_b, bin_c;
            bin_a.f = a[idx];
            bin_b.f = b[idx];
            bin_c.u = bin_a.u ^ bin_b.u;
            c[idx] = bin_c.f;
        }
    }
}

__global__ void float_bitwiseXOR_kernel_j(float* c, const float* a, const float* b, const int N, const int J) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < J; j++) {
        if (idx + j < N) {
            bin_float bin_a, bin_b, bin_c;
            bin_a.f = a[idx + j];
            bin_b.f = b[idx + j];
            bin_c.u = bin_a.u ^ bin_b.u;
            c[idx + j] = bin_c.f;
        }
    }
}

__global__ void float_bitwiseXOR_kernel(float* c, const float* a, const float* b, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        bin_float bin_a, bin_b, bin_c;
        bin_a.f = a[idx];
        bin_b.f = b[idx];
        bin_c.u = bin_a.u ^ bin_b.u;
        c[idx] = bin_c.f;
    }
}

float probe_kernel_mono(int array_size, int thread_nb, Metric metric_choice, int nb_iterations, int J, int K) {

    float
        //GPU arrays
        * dev_a = 0, * dev_b = 0, * dev_c = 0,
        //host arrays
        host_a[array_size], host_b[array_size], host_c[array_size],
        //array of time
        gpu_runtimes[nb_iterations];
    int group_thread_nb = div_up(thread_nb, J);


    dim3 block_size(div_up(array_size, thread_nb));
    dim3 thread_size(group_thread_nb);

    cudaEvent_t start_gpu, end_gpu;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    CHK(cudaSetDevice(0));

    CHK(cudaMalloc((void**)&dev_c, array_size * sizeof(float)));
    CHK(cudaMalloc((void**)&dev_a, array_size * sizeof(float)));
    CHK(cudaMalloc((void**)&dev_b, array_size * sizeof(float)));

    for (int iter = 0; iter < nb_iterations; iter++) {

        for (int i = 0; i < array_size; i++) {
            host_a[i] = rand();
            host_b[i] = rand();
        }

        CHK(cudaMemcpy(dev_a, host_a, array_size * sizeof(float), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dev_b, host_b, array_size * sizeof(float), cudaMemcpyHostToDevice));
        if (J > 1) {
            cudaEventRecord(start_gpu);
            float_bitwiseXOR_kernel_j << <block_size, thread_size >> > (dev_c, dev_a, dev_b, array_size, J);
            cudaEventRecord(end_gpu);
        }
        else if (K > 1) {
            cudaEventRecord(start_gpu);
            float_bitwiseXOR_kernel_k << <block_size, thread_size >> > (dev_c, dev_a, dev_b, array_size, K);
            cudaEventRecord(end_gpu);
        }
        else {
            cudaEventRecord(start_gpu);
            float_bitwiseXOR_kernel << <block_size, thread_size >> > (dev_c, dev_a, dev_b, array_size);
            cudaEventRecord(end_gpu);
        }

        CHK(cudaGetLastError());

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(host_c, dev_c, array_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Make sure the stop_gpu event is recorded before doing the time computation
        CHK(cudaEventSynchronize(end_gpu));
        CHK(cudaEventElapsedTime(&gpu_runtimes[iter], start_gpu, end_gpu));
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
    }

    return compute_metric(metric_choice, gpu_runtimes, nb_iterations);

}

void benchmark_varsize(
    int max_size,
    int steps,
    int thread_size,
    Metric choice,
    int nb_iter,
    const char* filename
) {

    DataPoint data[max_size];

    for (int i = 1; i < max_size; i += steps) {
        data[i] = DataPoint(probe_kernel_mono(i, thread_size, choice, nb_iter, DEFAULT_J, DEFAULT_K), i);
    }
    const char* renamed_filename = std::string(filename).append("_varsize.csv").data();
    save_data(renamed_filename, data, max_size);
}

void benchmark_varj(
    int J,
    Metric choice,
    int nb_iter,
    const char* filename
) {

    DataPoint data[J];

    for (int j = 1; j < J; j++) {
        data[j] = DataPoint(probe_kernel_mono(DEFAULT_ARRAY_SIZE, DEFAULT_THREAD_SIZE, choice, nb_iter, j, DEFAULT_K), j);
    }
    const char* renamed_filename = std::string(filename).append("_varj.csv").data();
    save_data(renamed_filename, data, J);
}

void benchmark_vark(
    int K,
    Metric choice,
    int nb_iter,
    const char* filename
) {

    DataPoint data[K];

    for (int k = 1; k < K; k++) {
        data[k] = DataPoint(probe_kernel_mono(DEFAULT_ARRAY_SIZE, DEFAULT_THREAD_SIZE, choice, nb_iter, DEFAULT_J, k), k);
    }
    const char* renamed_filename = std::string(filename).append("_vark.csv").data();
    save_data(renamed_filename, data, K);
}


int main(int argc, char** argv) {


    CLI::App app{ "Float Bitwise XOR" };
    bool variadic_size;
    app.add_flag("-1, --var-size", variadic_size, "Use a variadic array size (default is false)");
    bool variadic_j;
    app.add_flag("-2, --var-j", variadic_j, "Use a variadic J value (default is false)");
    bool artificial_load;
    app.add_flag("-3, --artificial-load", artificial_load, "Use an artificial load in the kernel (default is false)");


    int pre_array_size{ 1 };
    app.add_option("-n, --number", pre_array_size, "upper bound on the array size(default is 1)")->check(CLI::PositiveNumber);

    int step_size{ 1 };
    app.add_option("-s, --step", step_size, "step size for array size (default is 1)")->check(CLI::PositiveNumber);

    int thread_size{ 1024 };
    app.add_option("-t, --thread", thread_size, "number of threads per block (default is 1024)")->check(CLI::PositiveNumber);
    // Metric metric_choice = Metric::avg;
    // app.add_option("-m, --Metric", metric_choice, "Metric to use for performance measurement (avg or median)")->check(CLI::IsMember({ "avg", "median" }));

    int num_iterations{ 1 };
    app.add_option("-i, --iterations", num_iterations, "number of iterations to run for performance measurement");

    int J{ 1 };
    app.add_option("-j, --J", J, "Size of the number of operations to perform in the kernel (default is 1)")->check(CLI::PositiveNumber);

    int K{ 1 };
    app.add_option("-k, --K", K, "Size of the number of operations to perform in the kernel (default is 1)")->check(CLI::PositiveNumber);

    std::string output_filename = "float_output";
    app.add_option("-o, --output", output_filename, "output file name for performance results");

    CLI11_PARSE(app, argc, argv);

    if (variadic_size) {
        benchmark_varsize(pre_array_size, step_size, thread_size, Metric::avg, num_iterations, output_filename.data());
    }
    else if (variadic_j) {
        benchmark_varj(J, Metric::avg, num_iterations, output_filename.data());
    }
    else if (artificial_load) {
        benchmark_vark(K, Metric::avg, num_iterations, output_filename.data());
    }
    return 0;
}