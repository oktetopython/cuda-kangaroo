/**
 * @file CudaErrorHandler.h
 * @brief Comprehensive CUDA error handling system
 * 
 * Provides unified, exception-safe CUDA error handling with detailed
 * context information and automatic resource cleanup.
 */

#ifndef CUDA_ERROR_HANDLER_H
#define CUDA_ERROR_HANDLER_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <atomic>

/**
 * @brief CUDA-specific exception class
 * 
 * Provides detailed error information including CUDA error code,
 * operation context, and file/line information.
 */
class CudaException : public std::runtime_error {
private:
    cudaError_t error_code_;
    std::string operation_;
    std::string file_;
    int line_;
    int device_id_;

public:
    /**
     * @brief Constructor with full context information
     * @param error CUDA error code
     * @param operation Description of failed operation
     * @param file Source file where error occurred
     * @param line Line number where error occurred
     * @param device_id GPU device ID (optional)
     */
    CudaException(cudaError_t error, const std::string& operation, 
                  const std::string& file, int line, int device_id = -1)
        : std::runtime_error(formatMessage(error, operation, file, line, device_id))
        , error_code_(error)
        , operation_(operation)
        , file_(file)
        , line_(line)
        , device_id_(device_id) {}

    /**
     * @brief Get CUDA error code
     * @return CUDA error code
     */
    cudaError_t getErrorCode() const noexcept { return error_code_; }

    /**
     * @brief Get operation description
     * @return Operation that failed
     */
    const std::string& getOperation() const noexcept { return operation_; }

    /**
     * @brief Get source file
     * @return Source file where error occurred
     */
    const std::string& getFile() const noexcept { return file_; }

    /**
     * @brief Get line number
     * @return Line number where error occurred
     */
    int getLine() const noexcept { return line_; }

    /**
     * @brief Get device ID
     * @return GPU device ID (-1 if not specified)
     */
    int getDeviceId() const noexcept { return device_id_; }

private:
    /**
     * @brief Format detailed error message
     */
    static std::string formatMessage(cudaError_t error, const std::string& operation,
                                   const std::string& file, int line, int device_id) {
        std::ostringstream oss;
        oss << "CUDA Error: " << cudaGetErrorString(error) 
            << " (Code: " << static_cast<int>(error) << ")\n"
            << "Operation: " << operation << "\n"
            << "Location: " << file << ":" << line;
        
        if (device_id >= 0) {
            oss << "\nDevice ID: " << device_id;
        }
        
        return oss.str();
    }
};

/**
 * @brief CUDA error checking with context information
 * 
 * This class provides comprehensive error checking for CUDA operations
 * with automatic context capture and detailed error reporting.
 */
class CudaErrorChecker {
private:
    static std::atomic<bool> error_checking_enabled_;
    static std::atomic<int> error_count_;

public:
    /**
     * @brief Check CUDA error with full context
     * @param error CUDA error code to check
     * @param operation Description of the operation
     * @param file Source file name
     * @param line Line number
     * @param device_id GPU device ID (optional)
     * @throws CudaException if error is not cudaSuccess
     */
    static void checkError(cudaError_t error, const std::string& operation,
                          const std::string& file, int line, int device_id = -1) {
        if (error != cudaSuccess) {
            error_count_.fetch_add(1, std::memory_order_relaxed);
            
            if (error_checking_enabled_.load(std::memory_order_acquire)) {
                throw CudaException(error, operation, file, line, device_id);
            }
        }
    }

    /**
     * @brief Check last CUDA error
     * @param operation Description of the operation
     * @param file Source file name
     * @param line Line number
     * @param device_id GPU device ID (optional)
     * @throws CudaException if there was an error
     */
    static void checkLastError(const std::string& operation,
                              const std::string& file, int line, int device_id = -1) {
        cudaError_t error = cudaGetLastError();
        checkError(error, operation, file, line, device_id);
    }

    /**
     * @brief Enable/disable error checking
     * @param enabled Whether to enable error checking
     */
    static void setErrorCheckingEnabled(bool enabled) noexcept {
        error_checking_enabled_.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Check if error checking is enabled
     * @return true if error checking is enabled
     */
    static bool isErrorCheckingEnabled() noexcept {
        return error_checking_enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get total error count
     * @return Number of CUDA errors encountered
     */
    static int getErrorCount() noexcept {
        return error_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset error count
     */
    static void resetErrorCount() noexcept {
        error_count_.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief Get device properties safely
     * @param device_id GPU device ID
     * @return Device properties
     * @throws CudaException on error
     */
    static cudaDeviceProp getDeviceProperties(int device_id) {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
        checkError(error, "cudaGetDeviceProperties", __FILE__, __LINE__, device_id);
        return prop;
    }

    /**
     * @brief Set device safely
     * @param device_id GPU device ID to set
     * @throws CudaException on error
     */
    static void setDevice(int device_id) {
        cudaError_t error = cudaSetDevice(device_id);
        checkError(error, "cudaSetDevice", __FILE__, __LINE__, device_id);
    }

    /**
     * @brief Synchronize device safely
     * @throws CudaException on error
     */
    static void deviceSynchronize() {
        int device_id = -1;
        cudaGetDevice(&device_id);
        
        cudaError_t error = cudaDeviceSynchronize();
        checkError(error, "cudaDeviceSynchronize", __FILE__, __LINE__, device_id);
    }
};

// Static member definitions
std::atomic<bool> CudaErrorChecker::error_checking_enabled_{true};
std::atomic<int> CudaErrorChecker::error_count_{0};

/**
 * @brief RAII CUDA memory management
 * 
 * Provides automatic cleanup of CUDA device memory with exception safety.
 */
template<typename T>
class CudaMemoryGuard {
private:
    T* device_ptr_;
    size_t size_;
    int device_id_;

public:
    /**
     * @brief Constructor - allocates device memory
     * @param size Number of elements to allocate
     * @param device_id GPU device ID (optional)
     * @throws CudaException on allocation failure
     */
    explicit CudaMemoryGuard(size_t size, int device_id = -1) 
        : device_ptr_(nullptr), size_(size), device_id_(device_id) {
        
        if (device_id >= 0) {
            CudaErrorChecker::setDevice(device_id);
        } else {
            cudaGetDevice(&device_id_);
        }
        
        cudaError_t error = cudaMalloc(&device_ptr_, size * sizeof(T));
        CudaErrorChecker::checkError(error, "cudaMalloc", __FILE__, __LINE__, device_id_);
    }

    /**
     * @brief Destructor - automatic cleanup
     */
    ~CudaMemoryGuard() {
        if (device_ptr_) {
            // Don't throw in destructor - just log error
            cudaError_t error = cudaFree(device_ptr_);
            if (error != cudaSuccess) {
                // Log error but don't throw
                fprintf(stderr, "Warning: cudaFree failed in destructor: %s\n", 
                       cudaGetErrorString(error));
            }
        }
    }

    // Non-copyable, movable
    CudaMemoryGuard(const CudaMemoryGuard&) = delete;
    CudaMemoryGuard& operator=(const CudaMemoryGuard&) = delete;
    
    CudaMemoryGuard(CudaMemoryGuard&& other) noexcept
        : device_ptr_(other.device_ptr_), size_(other.size_), device_id_(other.device_id_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaMemoryGuard& operator=(CudaMemoryGuard&& other) noexcept {
        if (this != &other) {
            if (device_ptr_) {
                cudaFree(device_ptr_);
            }
            device_ptr_ = other.device_ptr_;
            size_ = other.size_;
            device_id_ = other.device_id_;
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get device pointer
     * @return Device memory pointer
     */
    T* get() noexcept { return device_ptr_; }

    /**
     * @brief Get const device pointer
     * @return Const device memory pointer
     */
    const T* get() const noexcept { return device_ptr_; }

    /**
     * @brief Get allocated size
     * @return Number of elements allocated
     */
    size_t size() const noexcept { return size_; }

    /**
     * @brief Get device ID
     * @return GPU device ID
     */
    int deviceId() const noexcept { return device_id_; }

    /**
     * @brief Copy data to device
     * @param host_data Host data to copy
     * @param count Number of elements to copy
     * @throws CudaException on error
     */
    void copyFromHost(const T* host_data, size_t count) {
        if (count > size_) {
            throw std::invalid_argument("Copy count exceeds allocated size");
        }
        
        cudaError_t error = cudaMemcpy(device_ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        CudaErrorChecker::checkError(error, "cudaMemcpy (H2D)", __FILE__, __LINE__, device_id_);
    }

    /**
     * @brief Copy data from device
     * @param host_data Host buffer to copy to
     * @param count Number of elements to copy
     * @throws CudaException on error
     */
    void copyToHost(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::invalid_argument("Copy count exceeds allocated size");
        }
        
        cudaError_t error = cudaMemcpy(host_data, device_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        CudaErrorChecker::checkError(error, "cudaMemcpy (D2H)", __FILE__, __LINE__, device_id_);
    }
};

/**
 * @brief Convenience macros for CUDA error checking
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        CudaErrorChecker::checkError(error, #call, __FILE__, __LINE__); \
    } while(0)

#define CUDA_CHECK_LAST_ERROR(operation) \
    CudaErrorChecker::checkLastError(operation, __FILE__, __LINE__)

#define CUDA_CHECK_DEVICE(call, device_id) \
    do { \
        cudaError_t error = (call); \
        CudaErrorChecker::checkError(error, #call, __FILE__, __LINE__, device_id); \
    } while(0)

/**
 * @brief CUDA kernel launch wrapper with error checking
 */
#define CUDA_LAUNCH_KERNEL(kernel, grid, block, shared_mem, stream, ...) \
    do { \
        kernel<<<grid, block, shared_mem, stream>>>(__VA_ARGS__); \
        CUDA_CHECK_LAST_ERROR("Kernel launch: " #kernel); \
    } while(0)

#endif // CUDA_ERROR_HANDLER_H