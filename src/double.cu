#include <bit>
#include <chrono>
#include "includes/commons.hpp"

//conversion double <-> binary
union bin_double {
    double value;
    u_int64_t binary;
};

/**
 * Compute binary operations as doubles (1 double/thread)
 */
__global__ void xor_double(bin_double* res, bin_double* left, bin_double* right, size_t N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        res[idx].binary = left[idx].binary ^ right[idx].binary;
    }
}

bool check_values(bin_double* cpu_res, bin_double* gpu_res, size_t N) {
    for (int i = 0; i < N; i++) {
        if (cpu_res[i].value != gpu_res[i].value)
            return false;
    }
    return true;
}

int bench_mono(float* gpu_time, const int max_size);

int main() {

    //TODO: make this as args
    const char* file_name = "results/output-double.csv";
    int max_size = 20;

    float time_array[max_size];

    bench_mono(time_array, max_size);
    save_data(file_name, time_array, max_size);
}

int bench_mono(float* gpu_time, const int max_size) {

    bin_double
        //inputs
        a[max_size], b[max_size],
        //outputs
        cpu_res[max_size], gpu_res[max_size],
        //GPU pointers
        * d_left, * d_right, * d_res;

    //fill arrays of random values
    for (int i = 0; i < max_size; i++) {
        a[i].value = static_cast<double>(rand());
        b[i].value = static_cast<double>(rand());
    }

    //event used to record time on GPU
    cudaEvent_t start_gpu, stop_gpu;

    /* CPU part */
    for (int i = 0; i < max_size; i++) {
        cpu_res[i].binary = a[i].binary ^ b[i].binary;
    }

    /* GPU part */

    //init GPU
    CHK(cudaSetDevice(0));

    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    //malloc vars
    CHK(cudaMalloc((void**)&d_left, max_size * sizeof(bin_double)));
    CHK(cudaMalloc((void**)&d_right, max_size * sizeof(bin_double)));
    CHK(cudaMalloc((void**)&d_res, max_size * sizeof(bin_double)));

    //Benchmark
    for (int array_size = 1; array_size < max_size; array_size++) {

        //threads & block sizes
        int thread_size(1024),
            block_size = div_up(array_size, thread_size);

        size_t vec_size = array_size * sizeof(bin_double);

        //clear result array in GPU
        CHK(cudaMemset(d_res, 0, vec_size));

        //load data for operations
        CHK(cudaMemcpy(d_left, a, vec_size, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_right, b, vec_size, cudaMemcpyHostToDevice));

        //run kernel
        CHK(cudaEventRecord(start_gpu));
        xor_double << <block_size, thread_size >> > (d_res, d_left, d_right, array_size);
        CHK(cudaEventRecord(stop_gpu));

        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());

        CHK(cudaMemcpy(gpu_res, d_res, array_size * sizeof(bin_double), cudaMemcpyDeviceToHost));

        //check results: discard not valid runs
        if (!check_values(cpu_res, gpu_res, array_size)) {
            printf("Result do not match !");
            continue;
        }

        //save time for this run
        CHK(cudaEventElapsedTime(&gpu_time[array_size], start_gpu, stop_gpu));
    }

Error:
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_res);

    cudaError_t status = cudaDeviceReset();
    if (status != cudaSuccess) {
        fprintf(stderr, "Could not reset device !");
        return 1;
    }
    return 0;
}