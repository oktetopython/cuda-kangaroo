cmake_minimum_required(VERSION 3.18)

# Try to find CUDA toolkit (optional)
find_package(CUDAToolkit QUIET)

# Automatically determine GPU support based on CUDA availability
if(CUDAToolkit_FOUND)
    set(WITHGPU ON)
    project(Kangaroo LANGUAGES CXX CUDA)
    message(STATUS "CUDA Toolkit found - enabling GPU support")
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")

    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    set(WITHGPU OFF)
    project(Kangaroo LANGUAGES CXX)
    message(STATUS "CUDA Toolkit not found - building CPU-only version")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific settings
if(WIN32)
    if(WITHGPU)
        add_definitions(-DWIN64 -D_CRT_SECURE_NO_WARNINGS -DWITHGPU -DUSE_MODERN_CUDA -DAUTO_GPU_DETECT)
    else()
        add_definitions(-DWIN64 -D_CRT_SECURE_NO_WARNINGS)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /wd4819")
else()
    if(WITHGPU)
        add_definitions(-DLINUX -DWITHGPU -DUSE_MODERN_CUDA -DAUTO_GPU_DETECT)
    else()
        add_definitions(-DLINUX)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# CUDA architectures - 根据CUDA版本自动调整
# CUDA architecture settings (only if GPU support is enabled)
if(WITHGPU)
    # Comprehensive GPU architecture support from Maxwell to Hopper
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
        # CUDA 12.0+ supports latest architectures including Hopper
        set(CMAKE_CUDA_ARCHITECTURES "52;53;60;61;70;72;75;80;86;89;90")
        message(STATUS "CUDA 12.0+ detected - enabling full architecture support (sm_52 to sm_90)")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
        # CUDA 11.8+ supports Ada Lovelace
        set(CMAKE_CUDA_ARCHITECTURES "52;53;60;61;70;72;75;80;86;89")
        message(STATUS "CUDA 11.8+ detected - enabling architecture support (sm_52 to sm_89)")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
        # CUDA 11.x supports up to Ampere architecture
        set(CMAKE_CUDA_ARCHITECTURES "52;53;60;61;70;72;75;80;86")
        message(STATUS "CUDA 11.x detected - enabling architecture support (sm_52 to sm_86)")
    else()
        # CUDA 10.x and earlier versions
        set(CMAKE_CUDA_ARCHITECTURES "52;53;60;61;70;72;75")
        message(STATUS "CUDA 10.x detected - enabling architecture support (sm_52 to sm_75)")
    endif()

    # Display supported architectures
    message(STATUS "Supported GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Include directories
include_directories(. SECPK1 GPU optimizations/phase4)

# SECPK1 Library Sources (shared across all executables)
set(SECPK1_SOURCES
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Point.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

# Create SECPK1 static library to eliminate code duplication
add_library(secpk1 STATIC ${SECPK1_SOURCES})
target_include_directories(secpk1 PUBLIC SECPK1)

# Optimized Checkpoint System Sources
set(CHECKPOINT_SOURCES
    OptimizedCheckpoint.cpp
    CheckpointUtils.cpp
    KangarooCheckpointIntegration.cpp
)

# Create optimized checkpoint static library
add_library(optimized_checkpoint STATIC ${CHECKPOINT_SOURCES})
if(WITHGPU)
    target_include_directories(optimized_checkpoint PUBLIC . ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(optimized_checkpoint secpk1 CUDA::cudart)
else()
    target_include_directories(optimized_checkpoint PUBLIC .)
    target_link_libraries(optimized_checkpoint secpk1)
endif()

# Kangaroo Core Sources (specific to main program)
set(KANGAROO_CORE_SOURCES
    main.cpp
    Kangaroo.cpp
    CommonUtils.cpp
    HashTable.cpp
    Thread.cpp
    Merge.cpp
    PartMerge.cpp
    Network.cpp
    Backup.cpp
    Check.cpp
    SmartAllocator.cpp
    UnifiedErrorHandler.cpp
)

# Add GPU sources only if GPU support is enabled
if(WITHGPU)
    list(APPEND KANGAROO_CORE_SOURCES
        GPU/GPUEngine.cu
        GPU/GPUGenerate.cpp
    )
endif()

# Create main executable
add_executable(kangaroo ${KANGAROO_CORE_SOURCES})
target_link_libraries(kangaroo optimized_checkpoint secpk1)

# Bitcoin Puzzle 135 Challenge System
add_executable(puzzle135_challenge
    puzzle135_challenge.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

# Bitcoin Puzzle 135 Table Generator
add_executable(puzzle135_bl_generator
    puzzle135_bl_generator.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

# Bitcoin Puzzle 135 System Verification
add_executable(test_puzzle135_system
    test_puzzle135_system.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

# Small Scale Algorithm Validation
add_executable(test_small_puzzle
    test_small_puzzle.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

# Additional test programs
add_executable(test_kangaroo_bl_integration
    test_kangaroo_bl_integration.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

add_executable(performance_benchmark
    performance_benchmark.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

# Real EC Table Generator
add_executable(generate_bl_real_ec_table
    optimizations/phase4/generate_bl_real_ec_table.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp
)

target_include_directories(generate_bl_real_ec_table PRIVATE SECPK1 optimizations/phase4)

# Additional development test programs (existing files)
add_executable(test_simple_solver
    test_simple_solver.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    SECPK1/IntGroup.cpp
    Timer.cpp
)

target_include_directories(test_simple_solver PRIVATE SECPK1)

add_executable(test_minimal_load
    test_minimal_load.cpp
)

add_executable(test_real_ec_solver
    test_real_ec_solver.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    SECPK1/IntGroup.cpp
    Timer.cpp
)

target_include_directories(test_real_ec_solver PRIVATE SECPK1)

# Bitcoin Puzzle 130 Test
add_executable(bitcoin_puzzle_130_test
    bitcoin_puzzle_130_test.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    SECPK1/IntGroup.cpp
    Timer.cpp
)

target_include_directories(bitcoin_puzzle_130_test PRIVATE SECPK1)

# CUDA properties (only if GPU support is enabled)
if(WITHGPU)
    set_target_properties(kangaroo PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Suppress CUDA header warnings on Windows
    if(WIN32)
        target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>)
    endif()
endif()

# Link libraries
if(WITHGPU)
    target_link_libraries(kangaroo secpk1 CUDA::cudart)
else()
    target_link_libraries(kangaroo secpk1)
endif()

# Platform-specific libraries
if(WIN32)
    target_link_libraries(kangaroo ws2_32)
endif()

# Compiler-specific flags
if(MSVC)
    target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3>)
    if(WITHGPU)
        target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -lineinfo>)
    endif()
else()
    target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
    if(WITHGPU)
        target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -lineinfo>)
    endif()
endif()

# Release configuration
set(CMAKE_BUILD_TYPE Release)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
        if(WITHGPU)
            target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        endif()
    else()
        target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
        if(WITHGPU)
            target_compile_options(kangaroo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        endif()
    endif()
endif()

# Note: Bitcoin Puzzle 135 components already defined above

# Add EC verification test
add_executable(ec_verification_test
    ec_verification_test.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    SECPK1/IntGroup.cpp
    Timer.cpp
)

target_include_directories(ec_verification_test PRIVATE SECPK1)

# Add BL algorithm verification test
add_executable(bl_algorithm_verification
    bl_algorithm_verification.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    SECPK1/IntGroup.cpp
    Timer.cpp
)

target_include_directories(bl_algorithm_verification PRIVATE SECPK1 optimizations/phase4)

# Add debug point structure test
add_executable(debug_point_structure
    debug_point_structure.cpp
    SECPK1/SECP256K1.cpp
    SECPK1/Point.cpp
    SECPK1/Int.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    SECPK1/IntGroup.cpp
    Timer.cpp
)

target_include_directories(debug_point_structure PRIVATE SECPK1)

# Note: test_puzzle135_system and test_small_puzzle already defined above

# Add test executable for GPU memory fix
add_executable(test_gpu_memory_fix
    test_gpu_memory_fix.cpp
    ${SECPK1_SOURCES}
    GPU/GPUEngine.cu
    Timer.cpp
)
target_link_libraries(test_gpu_memory_fix ${CUDA_LIBRARIES} ${OPENSSL_LIBRARIES} ws2_32)
target_include_directories(test_gpu_memory_fix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/SECPK1 ${CMAKE_CURRENT_SOURCE_DIR}/GPU ${CMAKE_CURRENT_SOURCE_DIR}/optimizations/phase4)
if(CUDA_FOUND)
    set_property(TARGET test_gpu_memory_fix PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Add test executable for optimized checkpoint system
add_executable(test_optimized_checkpoint
    test_optimized_checkpoint.cpp
    HashTable.cpp
    ${SECPK1_SOURCES}
)
target_link_libraries(test_optimized_checkpoint optimized_checkpoint secpk1)
target_include_directories(test_optimized_checkpoint PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/SECPK1)

# Print configuration
message(STATUS "=== Kangaroo Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "=====================================")
