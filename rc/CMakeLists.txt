cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 17)
project(RC VERSION 1.0 LANGUAGES CXX)

if(NOT DEFINED ENV{LIBTORCH_PATH})
    message(FATAL_ERROR "LIBTORCH_PATH environment variable is not set. Please set it in your .bashrc (e.g., export LIBTORCH_PATH=/path/to/libtorch/share/cmake/Torch/).")
else()
    set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})
endif()

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} rc.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${TORCH_LIBRARIES}
)

