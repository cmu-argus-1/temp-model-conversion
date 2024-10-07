cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 17)

project(NMS VERSION 1.0 LANGUAGES CXX)

# Check if LIBTORCH_PATH is set in the environment
if(NOT DEFINED ENV{LIBTORCH_PATH})
    message(FATAL_ERROR "LIBTORCH_PATH environment variable is not set. Please set it in your .bashrc (e.g., export LIBTORCH_PATH=/path/to/libtorch/share/cmake/Torch/).")
else()
    set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})
endif()

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Include OpenCV directories
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})


# Add the executable
add_executable(${PROJECT_NAME} ex_nms.cpp)

# Link libraries to the executable
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)