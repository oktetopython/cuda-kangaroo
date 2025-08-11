/*
* CUDA Error Handling Utilities
* Unified error handling for all CUDA operations
*/

#ifndef CUDA_ERROR_HANDLER_H
#define CUDA_ERROR_HANDLER_H

#include <cuda_runtime.h>
#include <stdio.h>

// Unified CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        CudaErrorHandler::HandleError(err, #call, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

// CUDA error checking with custom message
#define CUDA_CHECK_MSG(call, msg) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        CudaErrorHandler::HandleErrorWithMessage(err, msg, #call, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

// CUDA error checking for void functions
#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        CudaErrorHandler::HandleError(err, #call, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// CUDA error checking with return value
#define CUDA_CHECK_RETURN(call, retval) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        CudaErrorHandler::HandleError(err, #call, __FILE__, __LINE__); \
        return retval; \
    } \
} while(0)

class CudaErrorHandler {
public:
    // Handle CUDA error with detailed information
    static void HandleError(cudaError_t err, const char* call, const char* file, int line) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("  Call: %s\n", call);
        printf("  File: %s:%d\n", file, line);
    }
    
    // Handle CUDA error with custom message
    static void HandleErrorWithMessage(cudaError_t err, const char* message, 
                                     const char* call, const char* file, int line) {
        printf("CUDA Error: %s\n", message);
        printf("  CUDA Status: %s\n", cudaGetErrorString(err));
        printf("  Call: %s\n", call);
        printf("  File: %s:%d\n", file, line);
    }
    
    // GPUEngine specific error handler
    static void HandleGPUEngineError(cudaError_t err, const char* operation) {
        printf("GPUEngine: %s: %s\n", operation, cudaGetErrorString(err));
    }
    
    // Memory allocation error handler
    static void HandleMemoryError(cudaError_t err, const char* memType, size_t size) {
        printf("GPUEngine: Failed to allocate %s memory (%zu bytes): %s\n", 
               memType, size, cudaGetErrorString(err));
    }
    
    // Kernel launch error handler
    static void HandleKernelError(cudaError_t err, const char* kernelName) {
        printf("GPUEngine: Kernel '%s' execution failed: %s\n", 
               kernelName, cudaGetErrorString(err));
    }
};

#endif // CUDA_ERROR_HANDLER_H