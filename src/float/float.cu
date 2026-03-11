#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "includes/commons.h"
#include "includes/CLI11.hpp"

using std::rand;

union bin_float {
    float f;
    u_int32_t u;
};


bool checkCuda(int* out_cpu, int* out_gpu, int N);


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

float probe_kernel_mono(int array_size, int thread_nb, metric metric_choice, int nb_iterations){
    
    float 
        //GPU arrays
        *dev_a = 0, * dev_b = 0, * dev_c = 0,
        //host arrays
        host_a[array_size], host_b[array_size], host_c[array_size],
        //array of time
        gpu_runtimes[nb_iterations];

    dim3 block_size(div_up(array_size,thread_nb));
    dim3 thread_size(thread_nb);

    cudaEvent_t start_gpu, end_gpu;

    float results = 0;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    CHK(cudaSetDevice(0));

    CHK(cudaMalloc((void**)&dev_c, array_size * sizeof(float)));
    CHK(cudaMalloc((void**)&dev_a, array_size * sizeof(float)));
    CHK(cudaMalloc((void**)&dev_b, array_size * sizeof(float)));

    for(int iter = 0; iter < nb_iterations; iter++){

        for (int i = 0; i < array_size; i++) {
            host_a[i] = rand();
            host_b[i] = rand();
        }

        CHK(cudaMemcpy(dev_a, host_a, array_size * sizeof(float), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dev_b, host_b, array_size * sizeof(float), cudaMemcpyHostToDevice));

        cudaEventRecord(start_gpu);
        float_bitwiseXOR_kernel<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, array_size);
        cudaEventRecord(end_gpu);

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

    return compute_metric(metric_choice,gpu_runtimes,nb_iterations);

}

void benchmark_mono(
    int max_size, 
    int steps, 
    int thread_size, 
    metric choice, 
    int nb_iter, 
    const char* filename
){

    float results[max_size];

    for(int i = 1; i < max_size; i+=steps){
        //benchmark_kernel
        results[i] = probe_kernel_mono(i,thread_size,choice,nb_iter);
    }

    save_data(filename,results,max_size);
}