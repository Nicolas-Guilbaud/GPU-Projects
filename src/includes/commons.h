#pragma once

#include <stdio.h>
#include <fstream>

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

int div_up(int a, int b){
    return (a + b - 1)/b;
}

enum class metric {
    avg,
    mean,
};

void save_data(const char* filename, const float* time, int N){
    std::ofstream fout;
    fout.open(filename);
    if(!fout.good()){
        printf("Could not open %s !",filename);
    }

    for(int i = 1; i < N; i++){
        fout << i << "," << time[i] << "\n";
    }
    fout.close();

}

//FIXME: must be implemented
float compute_median(float* values, size_t nb_iter){
    return 0;
}

float compute_avg(float* values, size_t nb_iter){
    float res = 0;
    for(int i = 0; i < nb_iter; i++){
        res+=values[i];
    }
    return res/nb_iter;
}

float compute_metric(metric choice,float* values, size_t nb_iter){
    switch (choice){
        case metric::avg:
            return compute_avg(values,nb_iter);
        case metric::mean:
            return compute_median(values,nb_iter);
    }
}