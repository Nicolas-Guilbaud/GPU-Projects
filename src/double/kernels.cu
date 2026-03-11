#include <bit>
#include <chrono>
#include "includes/commons.hpp"

//conversion double <-> binary
union bin_double{
    double value;
    u_int64_t binary;
};

/**
 * Compute 1 binary operatio as double, 1 element/thread)
 */
__global__ void xor_double_mono(bin_double* res, bin_double* left, bin_double* right, size_t N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx < N){
        res[idx].binary = left[idx].binary ^ right[idx].binary;
    }
}

/**
 * Compute 1 binary operation as double, multiple elements/thread
 */
__global__ void xor_double_multiple(
    bin_double *res, 
    const bin_double *left, 
    const bin_double *right,
    size_t N,
    size_t nb_elem
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N){
        for(int j = 0; j < nb_elem; j++){
            res[idx+j].binary = left[idx+j].binary ^ right[idx+j].binary;
        }
    }
}

/**
 * Compute k times the binary operation as double, 1 element/thread
 */
__global__ void xor_double_repeated(
    bin_double *res,
    bin_double *left,
    bin_double *right,
    size_t N,
    int nb_repeats
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N){
        for(int k = 0; k < nb_repeats; k++){
            res[idx].binary = left[idx].binary ^ right[idx].binary;
        }
    }
}