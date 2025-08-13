cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libgecc LANGUAGES CUDA CXX)

include(GoogleTest)

enable_testing()
# Use parent project's googletest if available
if(TARGET gtest)
    message(STATUS "Using parent project's googletest")
    set(GTEST_FOUND TRUE)
else()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(WARNING "GTest not found, some gECC tests will be disabled")
        set(GTEST_FOUND FALSE)
    endif()
endif()

# find_package(CUDA 11.1 REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80")
endif()
set(CMAKE_CUDA_STANDARD "17")

macro(cuda_target_compile_options TARGET VERBOSE)
  if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    if (${VERBOSE})
      target_compile_options(${TARGET} PRIVATE --ptxas-options=-v;-lineinfo;-ftz=true)
    endif ()
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  else()
    message(FATAL_ERROR "Unknown CMAKE_CUDA_COMPILER_ID ${CMAKE_CUDA_COMPILER_ID}")
  endif()
endmacro()

include_directories(include)

# Generate constants
file(GLOB CONSTANTS_GENERATER_SCRIPTS
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/constants_generator.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/constants.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/ccgen.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/ec_ops.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/ec.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/field.py"
)
set(GENERATED_CONSTANTS_HEADER
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_constants.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_ops_cc_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/ec_ops_add_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/ec_ops_dbl_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_test_constants.h"
)
find_package(Python3 QUIET)
if(Python3_FOUND)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
else()
    find_program(PYTHON_EXECUTABLE python)
    if(NOT PYTHON_EXECUTABLE)
        message(WARNING "Python not found, using pre-generated constants")
        set(PYTHON_EXECUTABLE "")
    endif()
endif()

if(PYTHON_EXECUTABLE)
    add_custom_command(OUTPUT ${GENERATED_CONSTANTS_HEADER}
      COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/constants_generator.py" --out "${CMAKE_CURRENT_BINARY_DIR}/include"
      DEPENDS ${CONSTANTS_GENERATER_SCRIPTS}
      COMMENT "Generating Constants..."
      VERBATIM
    )
else()
    # Create dummy files if Python is not available
    foreach(header ${GENERATED_CONSTANTS_HEADER})
        add_custom_command(OUTPUT ${header}
            COMMAND ${CMAKE_COMMAND} -E touch ${header}
            COMMENT "Creating dummy ${header} (Python not available)"
        )
    endforeach()
endif()
add_custom_target(generated_constants DEPENDS ${GENERATED_CONSTANTS_HEADER})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Only add tests if GTest is properly available
if(GTEST_FOUND AND TARGET gtest)
    add_subdirectory(test)
else()
    message(STATUS "Skipping gECC tests - GTest not properly configured")
endif()

# Create a library target for gECC
add_library(libgecc INTERFACE)
target_include_directories(libgecc INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
add_dependencies(libgecc generated_constants)
