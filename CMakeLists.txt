cmake_minimum_required(VERSION 3.15)
project(GPU_Architecture_Simulator VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create main executable
add_executable(gpu_simulator ${SOURCES} ${HEADERS})

# Test executable
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
if(TEST_SOURCES)
    add_executable(gpu_tests ${TEST_SOURCES} 
                   src/gpu_core.cpp 
                   src/memory_hierarchy.cpp 
                   src/graphics_pipeline.cpp 
                   src/texture_cache.cpp 
                   src/performance_monitor.cpp)
    target_include_directories(gpu_tests PRIVATE include)
endif()

# Install targets
install(TARGETS gpu_simulator DESTINATION bin)