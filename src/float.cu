#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>  
#include <CLI/CLI.hpp>

using std::rand;

union bin_float {
    float f;
    u_int32_t u;
};


bool checkCuda(int* out_cpu, int* out_gpu, int N);
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

__global__ void float_bitwiseXOR_kernel(float* c, const float* a, const float* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        bin_float bin_a, bin_b, bin_c;
        bin_a.f = a[idx];
        bin_b.f = b[idx];
        bin_c.u = bin_a.u ^ bin_b.u;
        c[idx] = bin_c.f;
    }
}

enum class metric {
    avg,
    mean,
};

void main_kernel(int pre_array_size, int step_size, int tsize, metric metric_choice, int num_iterations, char* output_filename) {
    std::ofstream output_file(output_filename);
    for (int _array_size = 1; _array_size <= pre_array_size; _array_size += step_size) {
        std::cout << "Running with array size: " << _array_size << std::endl;
        const int array_size = _array_size;
        float* dev_a = 0, * dev_b = 0, * dev_c = 0;
        float host_a[array_size], host_b[array_size], host_c[array_size];
        dim3 block_size((array_size + tsize - 1) / tsize);
        dim3 thread_size(tsize);
        int iter;
        int i;
        cudaEvent_t start_gpu, end_gpu;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&end_gpu);
        float gpu_runtimes[num_iterations] = { 0 };
        CHK(cudaSetDevice(0));
        CHK(cudaMalloc((void**)&dev_c, array_size * sizeof(float)));
        CHK(cudaMalloc((void**)&dev_a, array_size * sizeof(float)));
        CHK(cudaMalloc((void**)&dev_b, array_size * sizeof(float)));
        for (iter = 0; iter < num_iterations; iter++) {
            for (i = 0; i < array_size; i++) {
                host_a[i] = rand();
                host_b[i] = rand();
            }


            CHK(cudaMemcpy(dev_a, host_a, array_size * sizeof(float), cudaMemcpyHostToDevice));
            CHK(cudaMemcpy(dev_b, host_b, array_size * sizeof(float), cudaMemcpyHostToDevice));

            cudaEventRecord(start_gpu);
            float_bitwiseXOR_kernel << <block_size, thread_size >> > (dev_c, dev_a, dev_b, array_size);
            cudaEventRecord(end_gpu);

            CHK(cudaGetLastError());

            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            CHK(cudaDeviceSynchronize());
            CHK(cudaMemcpy(host_c, dev_c, array_size * sizeof(float), cudaMemcpyDeviceToHost));

            // Make sure the stop_gpu event is recorded before doing the time computation
            CHK(cudaEventSynchronize(end_gpu));
            CHK(cudaEventElapsedTime(&gpu_runtimes[iter], start_gpu, end_gpu));

        Error:
            cudaFree(dev_c);
            cudaFree(dev_a);
            cudaFree(dev_b);

            // cudaDeviceReset must be called before exiting in order for profiling and
            // tracing tools such as Nsight and Visual Profiler to show complete traces.
            if (cudaDeviceReset() != cudaSuccess)
            {
                output_file << "cudaDeviceReset failed!\n";
                return;
            }

        }
        float avg_gpu_runtime = 0.0f;
        for (int iter = 0; iter < num_iterations; iter++) {
            avg_gpu_runtime += gpu_runtimes[iter];
        }
        avg_gpu_runtime /= num_iterations;
        output_file << array_size << "," << avg_gpu_runtime << "\n";
    }
    output_file.close();
}


int main(int argc, char** argv) {
    CLI::App app{ "Float Bitwise XOR" };
    int pre_array_size{ 1 };
    app.add_option("-n, --number", pre_array_size, "upper bound on the array size(default is 1)")->check(CLI::PositiveNumber);

    int step_size{ 1 };
    app.add_option("-s, --step", step_size, "step size for array size (default is 1)")->check(CLI::PositiveNumber);

    int thread_size{ 1024 };
    app.add_option("-t, --thread", thread_size, "number of threads per block (default is 1024)")->check(CLI::PositiveNumber);
    // metric metric_choice = metric::avg;
    // app.add_option("-m, --metric", metric_choice, "metric to use for performance measurement (avg or mean)")->check(CLI::IsMember({ "avg", "mean" }));

    int num_iterations{ 1 };
    app.add_option("-i, --iterations", num_iterations, "number of iterations to run for performance measurement");

    std::string output_filename = "float_output.txt";
    app.add_option("-o, --output", output_filename, "output file name for performance results");

    CLI11_PARSE(app, argc, argv);
    main_kernel(pre_array_size, step_size, thread_size, metric::avg, num_iterations, output_filename.data());
    return 0;
}