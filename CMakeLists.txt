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

# Enhancement modules shared across targets
set(ENHANCEMENT_SOURCES
    src/gpu_core.cpp
    src/memory_hierarchy.cpp
    src/graphics_pipeline.cpp
    src/texture_cache.cpp
    src/performance_monitor.cpp
    src/warp_scheduler.cpp
    src/tensor_core.cpp
    src/sector_cache.cpp
    src/dram_timing.cpp
    src/simulation_engine.cpp
)

# Create main executable
add_executable(gpu_simulator ${SOURCES} ${HEADERS})

# Existing test suite (test_suite.cpp only)
add_executable(gpu_tests
               tests/test_suite.cpp
               ${ENHANCEMENT_SOURCES})
target_include_directories(gpu_tests PRIVATE include)

# Validation suite (Pillar 4 – new enhancement tests)
add_executable(validation_suite
               tests/validation_suite.cpp
               ${ENHANCEMENT_SOURCES})
target_include_directories(validation_suite PRIVATE include)

# Example executable
add_executable(simple_example examples/simple_example.cpp
               ${ENHANCEMENT_SOURCES})
target_include_directories(simple_example PRIVATE include)

# Install targets
install(TARGETS gpu_simulator DESTINATION bin)