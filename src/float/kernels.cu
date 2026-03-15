union bin_float {
    float value;
    u_int32_t binary;
};

__global__ void float_bitwiseXOR_kernel_k(bin_float* c, const bin_float* a, const bin_float* b, const int N, const int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int k = 0; k < K; k++) {
            c[idx].binary = a[idx].binary ^ b[idx].binary;
        }
    }
}

__global__ void float_bitwiseXOR_kernel_j(bin_float *__restrict__ c, const bin_float *__restrict__ a, const bin_float *__restrict__ b, const int N, const int J) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < J && idx + j < N; ++j) {
        c[idx+j].binary = a[idx+j].binary ^ b[idx+j].binary;
    }
}

__global__ void float_bitwiseXOR_kernel(bin_float* c, const bin_float* a, const bin_float* b, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx].binary = a[idx].binary ^ b[idx].binary;
    }
}