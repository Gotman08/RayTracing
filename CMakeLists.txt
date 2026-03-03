cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ==============================================================================
# Project Configuration
# ==============================================================================

# Check if we only want to build tests (no CUDA required)
option(BUILD_TESTS_ONLY "Build only tests without CUDA" OFF)

if(BUILD_TESTS_ONLY)
    project(CUDARayTracer
        VERSION 1.0.0
        DESCRIPTION "High-Performance CUDA Ray Tracer for HPC Clusters (Romeo2025)"
        LANGUAGES CXX
    )
else()
    project(CUDARayTracer
        VERSION 1.0.0
        DESCRIPTION "High-Performance CUDA Ray Tracer for HPC Clusters (Romeo2025)"
        LANGUAGES CXX CUDA
    )
endif()

# ==============================================================================
# CMake Options
# ==============================================================================
option(BUILD_TESTS "Build unit tests" OFF)
option(ENABLE_FAST_MATH "Enable fast math optimizations" ON)
option(ENABLE_DEBUG_OUTPUT "Enable debug output" OFF)

# ==============================================================================
# C++ Standards
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# CUDA Configuration (only when not BUILD_TESTS_ONLY)
# ==============================================================================
if(NOT BUILD_TESTS_ONLY)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA Architecture Configuration
    # Romeo2025 GH200/H100: sm_90 (Hopper)
    # A100: sm_80 (Ampere)
    # V100: sm_70 (Volta)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "90")  # Default to H100
    endif()

    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)

    # CUDA Compiler Flags
    set(CUDA_NVCC_FLAGS_COMMON
        -lineinfo
        --expt-relaxed-constexpr
        --extended-lambda
        -Xcompiler=-Wall
    )

    if(ENABLE_FAST_MATH)
        list(APPEND CUDA_NVCC_FLAGS_COMMON --use_fast_math)
    endif()

    # Release flags
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

    # Debug flags
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

    # Include Directories (with CUDA)
    include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # Source Files
    set(RAYTRACER_SOURCES
        src/main.cu
    )

    # Executable Target
    add_executable(raytracer ${RAYTRACER_SOURCES})

    target_compile_options(raytracer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS_COMMON}>
    )

    set_target_properties(raytracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_link_libraries(raytracer
        CUDA::cudart
        CUDA::curand
    )

    if(ENABLE_DEBUG_OUTPUT)
        target_compile_definitions(raytracer PRIVATE DEBUG_OUTPUT=1)
    endif()

    # Install Rules
    install(TARGETS raytracer RUNTIME DESTINATION bin)
    install(DIRECTORY scenes/ DESTINATION share/raytracer/scenes)
endif()

# ==============================================================================
# Tests (can be built with or without CUDA)
# ==============================================================================
if(BUILD_TESTS OR BUILD_TESTS_ONLY)
    enable_testing()

    add_executable(raytracer_tests
        tests/test_main.cpp
        tests/test_vec3.cpp
        tests/test_ray.cpp
        tests/test_interval.cpp
        tests/test_sphere.cpp
        tests/test_plane.cpp
        tests/test_aabb.cpp
    )

    target_include_directories(raytracer_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external
    )

    target_compile_options(raytracer_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )

    set_target_properties(raytracer_tests PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    add_test(NAME unit_tests COMMAND raytracer_tests)

    message(STATUS "Tests: ENABLED")
endif()

# ==============================================================================
# Print Configuration Summary
# ==============================================================================
message(STATUS "")
message(STATUS "=== CUDA Ray Tracer Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
if(NOT BUILD_TESTS_ONLY)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(BUILD_TESTS OR BUILD_TESTS_ONLY)
    message(STATUS "Tests: ON")
endif()
message(STATUS "")
