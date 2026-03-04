#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>  

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
#define N_THREAD 1

int main(int argc, char** argv) {
    const int array_size = 1;
    float* dev_a = 0, * dev_b = 0, * dev_c = 0;
    float host_a[array_size], host_b[array_size], host_c[array_size];
    float gpu_runtime_ms = 0.0f;
    char filename[] = "float_output.txt";
    std::ofstream output_file(filename);

    dim3 block_size(array_size + (N_THREAD - 1) / N_THREAD);
    dim3 thread_size(N_THREAD);

    for (int i = 0; i < array_size; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    cudaError_t cuda_status;
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    CHK(cudaSetDevice(0));
    CHK(cudaMalloc((void**)&dev_c, array_size * sizeof(float)));
    CHK(cudaMalloc((void**)&dev_a, array_size * sizeof(float)));
    CHK(cudaMalloc((void**)&dev_b, array_size * sizeof(float)));

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
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&gpu_runtime_ms, start_gpu, end_gpu);
    output_file << "GPU runtime: " << gpu_runtime_ms << " ms\n";

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess)
    {
        output_file << "cudaDeviceReset failed!  Do you have a CUDA-capable GPU installed?\n";
    }

    output_file.close();
    return 0;

}