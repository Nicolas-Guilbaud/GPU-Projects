#include <bit>
#include <chrono>
#include "float/float.cu"
#include "double/double.cu"
#include "CLI11.hpp"

int main(int argc, char** argv) {


    CLI::App app{ "Bitwise XOR" };

    bool is_double = false,
        is_float = false;
    
    int max_array_size = 1,
        step_size = 1,
        step_j = 1,
        step_k = 1,
        thread_size = DEFAULT_THREAD_SIZE,
        num_iterations = DEFAULT_ITERATIONS,
        J = DEFAULT_J,
        K = DEFAULT_K;
        
    std::string output_filename = "output";
    Metric metric_choice = Metric::avg;
    
    app.add_option("-t, --thread", thread_size, "number of threads per block (default is 1024)")->check(CLI::PositiveNumber);
    app.add_option("-m, --Metric", metric_choice, "Metric to use for performance measurement (avg or median)")->check(CLI::IsMember({ "avg", "median" }));
    
    app.add_option("-n, --max_value", max_array_size, "upper bound on the array size(default is 1)")->check(CLI::PositiveNumber);
    app.add_option("-i, --iterations", num_iterations, "number of iterations to run for performance measurement");
    app.add_option("-j, --J", J, "Size of the number of operations to perform in the kernel (default is 1)")->check(CLI::PositiveNumber);
    app.add_option("-k, --K", K, "Size of the number of operations to perform in the kernel (default is 1)")->check(CLI::PositiveNumber);
    app.add_option("-o, --output", output_filename, "output file name for performance results");
    
    app.add_option("-s, --step", step_size, "step size for array size (default is 1)")->check(CLI::PositiveNumber);
    app.add_option("--step_j", step_j, "step size for j size (default is 1)")->check(CLI::PositiveNumber);
    app.add_option("--step_k", step_k, "step size for k size (default is 1)")->check(CLI::PositiveNumber);
    

    app.add_flag("-d, --double", is_double, "Run the benchmarks with the double data type");
    app.add_flag("-f, --float", is_float, "Run the benchmarks with the float data type");

    CLI11_PARSE(app, argc, argv);

    if(!is_float && !is_double){
        printf("Please choose a correct data type: -d | -f");
        return 1;
    }

    if(is_float){
        std::string float_filename = "results/float_";
        float_filename.append(output_filename);
        printf("File: %s\n",float_filename.c_str());
        if (max_array_size > 1) {
            benchmark_varsize_float(max_array_size, step_size, thread_size, metric_choice, num_iterations, float_filename);
        }
        if (J > 1) {
            benchmark_varj_float(J, step_j, thread_size, metric_choice, num_iterations, float_filename);
        }
        if (K > 1) {
            benchmark_vark_float(K, step_k, thread_size, metric_choice, num_iterations, float_filename);
        }
    }
    
    if(is_double){
        std::string double_filename = "results/double_";
        double_filename.append(output_filename);
        if (max_array_size > 1) {
            benchmark_varsize_double(max_array_size, step_size, thread_size, metric_choice, num_iterations, double_filename);
        }
        if (J > 1) {
            benchmark_varj_double(J, step_j, thread_size, metric_choice, num_iterations, double_filename);
        }
        if (K > 1) {
            benchmark_vark_double(K, step_k, thread_size, metric_choice, num_iterations, double_filename);
        }
    }
    return 0;
}