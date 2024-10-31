#New CMake build file for FireHorn
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
## sometinh fishy here
project(SIL VERSION 0.1 LANGUAGES C CXX)

# Set the build type to Debug
set(CMAKE_BUILD_TYPE Debug)


# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage -fprofile-arcs -ftest-coverage -fprofile-abs-path")


# Set the output directory for the build
#add_executable(main_stubs 2024_C_AV_RPI/main.cpp)

add_subdirectory(2024_C_AV_RPI)

