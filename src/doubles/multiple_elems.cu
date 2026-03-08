#include "../includes/commons.h"

__global__ void xor_double_multiple(
    bin_double *res, 
    const bin_double *left, 
    const bin_double *right,
    size_t stride,
    size_t N
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N){
        for(int i = 0; i < stride; i++){
            res[idx+i].binary = left[idx+i].binary ^ right[idx+i].binary;
        }
    }
}

float execute_kernel(
    int array_size,
    int nb_threads,
    int j,
    metric choice,
    int nb_iter
){

    bin_double 
        //data
        a[array_size],b[array_size],res[array_size],
        //GPU
        *dev_a,*dev_b,*dev_res;
    
    float gpu_time[nb_iter];

    // [j] [j] ... -> array_size = j*threads*blocks
    // 
    dim3 thread_size(div_up(nb_threads,j)),
        block_size(div_up(array_size,nb_threads));
    
    cudaEvent_t start, stop;
    size_t vec_size = array_size*sizeof(bin_double);

    CHK(cudaSetDevice(0));
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    //malloc vars
    CHK(cudaMalloc((void**) &dev_a,vec_size));
    CHK(cudaMalloc((void**) &dev_b,vec_size));
    CHK(cudaMalloc((void**) &dev_res,vec_size));
    
    for(int iter = 0; iter < nb_iter; iter++){

        //fill arrays of random values
        for(int i = 0; i < array_size; i++){
            a[i].value = static_cast<double>(rand());
            b[i].value = static_cast<double>(rand());
        }

        //clear result array in GPU
        CHK(cudaMemset(dev_res,0,vec_size));

        //load data for operations
        CHK(cudaMemcpy(dev_a, a, vec_size,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dev_b, b, vec_size,cudaMemcpyHostToDevice));

        //run kernel
        CHK(cudaEventRecord(start));
        xor_double_multiple<<<block_size,thread_size>>>(dev_res,dev_a,dev_b,j,array_size);
        CHK(cudaEventRecord(stop));

        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());

        CHK(cudaMemcpy(res,dev_res,array_size*sizeof(bin_double),cudaMemcpyDeviceToHost));

        //save time for this run
        CHK(cudaEventElapsedTime(&gpu_time[iter],start,stop));


    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    cudaError_t status = cudaDeviceReset();
    if(status != cudaSuccess){
        fprintf(stderr,"Could not reset device !");
    }

    return compute_metric(choice,gpu_time,nb_iter);
}

void benchmark_multiple(
    int array_size,
    int nb_threads,
    int max_elems,
    metric choice,
    int nb_iter,
    const char* filename
){

    float values[max_elems];

    for(int j = 1; j < max_elems; j++){
        values[j] = execute_kernel(array_size,nb_threads,j,choice,nb_iter);
    }

    save_data(filename,values,max_elems);

}