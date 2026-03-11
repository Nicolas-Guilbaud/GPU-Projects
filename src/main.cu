#include <bit>
#include <chrono>
#include "double/benchmark_mono.cu"
#include "double/multiple_elems.cu"
#include "CLI11.hpp"

int main(int argc, char** argv){

    int max_size = 1;
    int step_size= 1;
    int thread_size = 1024;
    int num_iterations = 1;
    data_type type = data_type::DOUBLE;
    metric metric_choice = metric::avg;

    std::string file_name = "./results/output-double.csv";

    CLI::App app{ "Bitwise XOR" };
    app.add_option("-n, --number", max_size, "upper bound on the array size(default is 1)")->check(CLI::PositiveNumber);
    app.add_option("-s, --step", step_size, "step size for array size (default is 1)")->check(CLI::PositiveNumber);
    app.add_option("-t, --thread", thread_size, "number of threads per block (default is 1024)")->check(CLI::PositiveNumber);
    app.add_option("-m, --metric", metric_choice, "metric to use for performance measurement (avg or median)")->check(CLI::IsMember({ "avg", "median" }));
    app.add_option("-i, --iterations", num_iterations, "number of iterations to run for performance measurement");
    app.add_option("-o, --output", file_name, "output file name for performance results");
    // FIXME !
    // app.add_option("-d","--data-type", type,"type of the data to run")->check(CLI::IsMember({"DOUBLE", "FLOAT"}));

    CLI11_PARSE(app, argc, argv);

    //TODO: make this as args
    //benchmark_mono(max_size,step_size,thread_size,metric::avg,num_iterations,file_name.data());
    benchmark_multiple(5000,thread_size,5000,metric::avg,5,"doubles-multiple.csv");

    //Float
    //benchmark_mono(max_size, step_size, thread_size, metric::avg, num_iterations, file_name.data());

    return 0;
}