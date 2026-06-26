cmake_minimum_required(VERSION 3.18)
project(mytests LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86) # NV3080Ti

enable_testing()

add_subdirectory(3rdparty/googletest)

# Collect all .cc files in current directory
file(GLOB_RECURSE TEST_SOURCES_OF_CXX CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

# Collect all .cu files in current directory
file(GLOB_RECURSE TEST_SOURCES_OF_CUDA CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_executable(mytests ${TEST_SOURCES_OF_CXX} ${TEST_SOURCES_OF_CUDA})

# Tell the 'mytests' target where to find its header files.
target_include_directories(mytests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
# Link GoogleTest
target_link_libraries(mytests PRIVATE gtest Threads::Threads)

# Enable automatic test discovery
include(GoogleTest)
gtest_discover_tests(mytests)
