union bin_float {
    float value;
    u_int32_t binary;
};

__global__ void float_bitwiseXOR_kernel_k(float* c, const float* a, const float* b, const int N, const int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int k = 0; k < K; k++) {
            bin_float bin_a, bin_b, bin_c;
            bin_a.value = a[idx];
            bin_b.value = b[idx];
            bin_c.binary = bin_a.binary ^ bin_b.binary;
            c[idx] = bin_c.value;
        }
    }
}

__global__ void float_bitwiseXOR_kernel_j(float* c, const float* a, const float* b, const int N, const int J) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < J && idx + j < N; ++j) {
        bin_float bin_a, bin_b, bin_c;
        bin_a.value = a[idx + j];
        bin_b.value = b[idx + j];
        bin_c.binary = bin_a.binary ^ bin_b.binary;
        c[idx + j] = bin_c.value;
    }
}

__global__ void float_bitwiseXOR_kernel(float* c, const float* a, const float* b, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        bin_float bin_a, bin_b, bin_c;
        bin_a.value = a[idx];
        bin_b.value = b[idx];
        bin_c.binary = bin_a.binary ^ bin_b.binary;
        c[idx] = bin_c.value;
    }
}