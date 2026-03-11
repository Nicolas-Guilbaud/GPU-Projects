cmake_minimum_required(VERSION 3.20)

project("Mini-CUDA" LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

#CLI
include_directories(BEFORE "extern")
include_directories(BEFORE "src")


file(GLOB FLOAT_SOURCES "src/main.cu")
add_executable("main" ${FLOAT_SOURCES})

# Warn when switch statements over enums are not exhaustive.
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wswitch;-Wswitch-enum>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wswitch,-Wswitch-enum>
)
