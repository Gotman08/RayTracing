cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ==============================================================================
# Project Configuration
# ==============================================================================

# Check if we only want to build tests (no CUDA required)
option(BUILD_TESTS_ONLY "Build only tests without CUDA" OFF)
option(BUILD_CPU_ONLY "Build CPU-only raytracer without CUDA" OFF)

if(BUILD_TESTS_ONLY OR BUILD_CPU_ONLY)
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
option(ENABLE_INTERACTIVE "Enable interactive rendering mode with GLFW/OpenGL" ON)

# ==============================================================================
# C++ Standards
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# CPU-Only Raytracer (OpenMP)
# ==============================================================================
if(BUILD_CPU_ONLY)
    find_package(OpenMP REQUIRED)

    include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external
    )

    add_executable(raytracer src/main_cpu.cpp)

    target_compile_options(raytracer PRIVATE -Wall -O3)
    target_compile_definitions(raytracer PRIVATE CPU_ONLY=1)

    target_link_libraries(raytracer OpenMP::OpenMP_CXX)

    if(ENABLE_FAST_MATH)
        target_compile_options(raytracer PRIVATE -ffast-math)
    endif()

    install(TARGETS raytracer RUNTIME DESTINATION bin)

    message(STATUS "CPU-Only Mode: ENABLED (OpenMP)")
endif()

# ==============================================================================
# CUDA Configuration (only when not BUILD_TESTS_ONLY and not BUILD_CPU_ONLY)
# ==============================================================================
if(NOT BUILD_TESTS_ONLY AND NOT BUILD_CPU_ONLY)
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

    # OpenMP for CPU rendering path
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(raytracer OpenMP::OpenMP_CXX)
        # Pass OpenMP flag to host compiler via nvcc
        target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
        message(STATUS "OpenMP: ENABLED (for --cpu mode)")
    endif()

    target_link_libraries(raytracer
        CUDA::cudart
        CUDA::curand
    )

    # ==============================================================================
    # Interactive Mode (GLFW + OpenGL)
    # ==============================================================================
    if(ENABLE_INTERACTIVE)
        find_package(OpenGL REQUIRED)
        find_package(glfw3 3.3 REQUIRED)
        find_package(GLEW REQUIRED)

        target_compile_definitions(raytracer PRIVATE ENABLE_INTERACTIVE=1)
        target_link_libraries(raytracer OpenGL::GL glfw GLEW::GLEW)

        message(STATUS "Interactive Mode: ENABLED (GLFW + OpenGL + GLEW)")
    else()
        message(STATUS "Interactive Mode: DISABLED")
    endif()

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
        tests/test_materials.cpp
        tests/test_camera.cpp
        tests/test_tone_mapping.cpp
        tests/test_bvh.cpp
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
message(STATUS "=== Ray Tracer Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
if(BUILD_CPU_ONLY)
    message(STATUS "Mode: CPU-only (OpenMP)")
elseif(NOT BUILD_TESTS_ONLY)
    message(STATUS "Mode: CUDA + CPU")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(BUILD_TESTS OR BUILD_TESTS_ONLY)
    message(STATUS "Tests: ON")
endif()
message(STATUS "")
