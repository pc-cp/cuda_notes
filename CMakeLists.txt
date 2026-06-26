cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# message("PATH is $ENV{PATH}")
project(learn-cuda LANGUAGES CXX)

# default cmake is Debug, now its Release
if (NOT CMAKE_BUILD_TYPE) 
	set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

# Add GoogleTest from external directory
# Second argument specifies where to put gtest build files (inside your build folder)
add_subdirectory(/Users/pengchen/repos/googletest ${CMAKE_BINARY_DIR}/googletest-build)

# Collect all .cc files in current directory
file(GLOB_RECURSE TEST_SOURCES_OF_CXX CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

# Collect all .cu files in current directory
file(GLOB_RECURSE TEST_SOURCES_OF_CUDA CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")


# Your test executable (with all .cc files)
add_executable(mytests ${TEST_SOURCES_OF_CXX} ${TEST_SOURCES_OF_CUDA})

# Tell the 'mytests' target where to find its header files.
target_include_directories(mytests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link GoogleTest
target_link_libraries(mytests PRIVATE gtest)

# Enable automatic test discovery
include(GoogleTest)
gtest_discover_tests(mytests)
