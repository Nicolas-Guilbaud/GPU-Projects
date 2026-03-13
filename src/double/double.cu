#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "includes/commons.hpp"
#include "double/kernels.cu"

using std::rand;

float probe_kernel_double(int array_size, int thread_nb, Metric metric_choice, int nb_iterations, int J, int K) {

    bin_double
        //GPU arrays
        * dev_a = 0, * dev_b = 0, * dev_c = 0,
        //host arrays
        host_a[array_size], host_b[array_size], host_c[array_size];
    
    //array of time
    float gpu_runtimes[nb_iterations];

    int group_thread_nb = div_up(thread_nb, J);
    size_t vec_size = array_size * sizeof(float);


    dim3 block_size(div_up(array_size, thread_nb));
    dim3 thread_size(group_thread_nb);

    cudaEvent_t start_gpu, end_gpu;

    int iter = 0;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    CHK(cudaSetDevice(0));

    CHK(cudaMalloc((void**)&dev_c, vec_size));
    CHK(cudaMalloc((void**)&dev_a, vec_size));
    CHK(cudaMalloc((void**)&dev_b, vec_size));

    do {

        for (int i = 0; i < array_size; i++) {
            host_a[i].value = static_cast<double>(rand());
            host_b[i].value = static_cast<double>(rand());
        }

        CHK(cudaMemcpy(dev_a, host_a, vec_size, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dev_b, host_b, vec_size, cudaMemcpyHostToDevice));
        if (J > 1) {
            cudaEventRecord(start_gpu);
            xor_double_multiple<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, array_size, J);
            cudaEventRecord(end_gpu);
        }
        else if (K > 1) {
            cudaEventRecord(start_gpu);
            xor_double_repeated<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, array_size, K);
            cudaEventRecord(end_gpu);
        }
        else {
            cudaEventRecord(start_gpu);
            xor_double_mono<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, array_size);
            cudaEventRecord(end_gpu);
        }

        CHK(cudaGetLastError());

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(host_c, dev_c, vec_size, cudaMemcpyDeviceToHost));

        // Make sure the stop_gpu event is recorded before doing the time computation
        CHK(cudaEventSynchronize(end_gpu));
        CHK(cudaEventElapsedTime(&gpu_runtimes[iter], start_gpu, end_gpu));

        //Ensure computational time is not negative
        if(gpu_runtimes[iter] >= 0.0f){
            iter++;
        }
    }while(iter < nb_iterations);

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

void benchmark_varsize_double(
    int max_size,
    int steps,
    int thread_size,
    Metric choice,
    int nb_iter,
    std::string filename
) {

    DataPoint data[max_size];

    for (int i = 1; i < max_size; i += steps) {
        data[i] = DataPoint(probe_kernel_double(i, thread_size, choice, nb_iter, DEFAULT_J, DEFAULT_K), i);
    }
    std::string renamed_filename = std::string(filename).append("_varsize.csv");
    save_data(renamed_filename, data, max_size);
}

void benchmark_varj_double(
    int J,
    int steps,
    int thread_size,
    Metric choice,
    int nb_iter,
    std::string filename
) {

    DataPoint data[J];

    for (int j = 1; j < J; j += steps) {
        data[j] = DataPoint(probe_kernel_double(DEFAULT_ARRAY_SIZE, thread_size, choice, nb_iter, j, DEFAULT_K), j);
    }
    std::string renamed_filename = std::string(filename).append("_varj.csv");
    save_data(renamed_filename, data, J);
}

void benchmark_vark_double(
    int K,
    int steps,
    int thread_size,
    Metric choice,
    int nb_iter,
    std::string filename
) {

    DataPoint data[K];

    for (int k = 1; k < K; k += steps) {
        data[k] = DataPoint(probe_kernel_double(DEFAULT_ARRAY_SIZE, thread_size, choice, nb_iter, DEFAULT_J, k), k);
    }
    std::string renamed_filename = std::string(filename).append("_vark.csv");
    save_data(renamed_filename, data, K);
}