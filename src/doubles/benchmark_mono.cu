#include <bit>
#include <chrono>
#include "../includes/commons.h"

/**
 * Compute binary operations as doubles (1 double/thread)
 */
__global__ void xor_double(bin_double* res, bin_double* left, bin_double* right, size_t N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx < N){
        res[idx].binary = left[idx].binary ^ right[idx].binary;
    }
}

/**
 * executes the kernel for a fixed array size and measure the time 
 * required to complete the computation.
 * 
 */
float probe_kernel_mono(int array_size, int thread_nb, metric choice, int nb_iterations){

    bin_double 
        //inputs & outputs
        a[array_size],b[array_size], gpu_res[array_size],
        //GPU pointers
        *d_left, *d_right, *d_res;

    float gpu_time[nb_iterations];

    //threads & block sizes
    dim3 thread_size(thread_nb), 
        block_size(div_up(array_size,thread_nb));

    //event used to record time on GPU
    cudaEvent_t start_gpu, stop_gpu;

    size_t vec_size = array_size*sizeof(bin_double);
    
    for(int iter = 0; iter < nb_iterations; iter++){
        //fill arrays of random values
        for(int i = 0; i < array_size; i++){
            a[i].value = static_cast<double>(rand());
            b[i].value = static_cast<double>(rand());
        }

        /* GPU part */

        //init GPU
        CHK(cudaSetDevice(0));

        CHK(cudaEventCreate(&start_gpu));
        CHK(cudaEventCreate(&stop_gpu));

        //malloc vars
        CHK(cudaMalloc((void**) &d_left,vec_size));
        CHK(cudaMalloc((void**) &d_right,vec_size));
        CHK(cudaMalloc((void**) &d_res,vec_size));

        //clear result array in GPU
        CHK(cudaMemset(d_res,0,vec_size));

        //load data for operations
        CHK(cudaMemcpy(d_left, a, vec_size,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_right, b, vec_size,cudaMemcpyHostToDevice));

        //run kernel
        CHK(cudaEventRecord(start_gpu));
        xor_double<<<block_size,thread_size>>>(d_res,d_left,d_right,array_size);
        CHK(cudaEventRecord(stop_gpu));

        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());

        CHK(cudaMemcpy(gpu_res,d_res,array_size*sizeof(bin_double),cudaMemcpyDeviceToHost));

        //save time for this run
        CHK(cudaEventElapsedTime(&gpu_time[iter],start_gpu,stop_gpu));
    }

Error:
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_res);

    cudaError_t status = cudaDeviceReset();
    if(status != cudaSuccess){
        fprintf(stderr,"Could not reset device !");
    }

    return compute_metric(choice,gpu_time,nb_iterations);
}

void benchmark_mono(
    int max_size, 
    int steps, 
    int thread_size, 
    metric choice, 
    int nb_iter, 
    const char* filename
){

    float time_array[max_size];
    for(int i = 1; i < max_size; i+=steps){
        time_array[i] = probe_kernel_mono(i,thread_size,choice,nb_iter);
    }
    save_data(filename,time_array,max_size);
}