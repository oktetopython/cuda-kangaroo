/**
 * @file ModernMemoryManager.h
 * @brief Modern C++ RAII-based memory management for Kangaroo project
 * 
 * This header provides type-safe, exception-safe memory management
 * using modern C++ RAII principles to replace C-style malloc/free.
 */

#ifndef MODERN_MEMORY_MANAGER_H
#define MODERN_MEMORY_MANAGER_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <atomic>

#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc on Windows
#endif

/**
 * @brief RAII wrapper for thread parameters
 * 
 * Provides automatic memory management for TH_PARAM arrays
 * with proper initialization and cleanup.
 */
template<typename T>
class RAIIArray {
private:
    std::unique_ptr<T[]> data_;
    size_t size_;

public:
    /**
     * @brief Constructor - allocates and zero-initializes array
     * @param size Number of elements to allocate
     * @throws std::bad_alloc if allocation fails
     */
    explicit RAIIArray(size_t size) : size_(size) {
        if (size == 0) {
            throw std::invalid_argument("Array size cannot be zero");
        }
        
        data_ = std::make_unique<T[]>(size);
        
        // Zero-initialize the array (equivalent to calloc behavior)
        std::memset(data_.get(), 0, size * sizeof(T));
    }

    /**
     * @brief Destructor - automatic cleanup (RAII)
     */
    ~RAIIArray() = default;

    // Non-copyable, movable
    RAIIArray(const RAIIArray&) = delete;
    RAIIArray& operator=(const RAIIArray&) = delete;
    RAIIArray(RAIIArray&&) = default;
    RAIIArray& operator=(RAIIArray&&) = default;

    /**
     * @brief Array access operator
     * @param index Array index
     * @return Reference to element at index
     * @throws std::out_of_range if index is invalid
     */
    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Array index out of bounds");
        }
        return data_[index];
    }

    /**
     * @brief Const array access operator
     * @param index Array index
     * @return Const reference to element at index
     * @throws std::out_of_range if index is invalid
     */
    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Array index out of bounds");
        }
        return data_[index];
    }

    /**
     * @brief Get raw pointer (for C API compatibility)
     * @return Raw pointer to array data
     */
    T* get() noexcept {
        return data_.get();
    }

    /**
     * @brief Get const raw pointer
     * @return Const raw pointer to array data
     */
    const T* get() const noexcept {
        return data_.get();
    }

    /**
     * @brief Get array size
     * @return Number of elements in array
     */
    size_t size() const noexcept {
        return size_;
    }

    /**
     * @brief Check if array is valid
     * @return true if array is allocated and non-empty
     */
    bool valid() const noexcept {
        return data_ != nullptr && size_ > 0;
    }
};

/**
 * @brief RAII wrapper for hash table entries
 * 
 * Manages dynamic allocation of hash table entries with
 * automatic cleanup and exception safety.
 */
class RAIIHashEntry {
private:
    void* entry_ptr_;
    size_t entry_size_;

public:
    /**
     * @brief Constructor - allocates entry
     * @param size Size of entry to allocate
     * @throws std::bad_alloc if allocation fails
     */
    explicit RAIIHashEntry(size_t size) : entry_size_(size) {
        if (size == 0) {
            throw std::invalid_argument("Entry size cannot be zero");
        }
        
#ifdef _WIN32
        entry_ptr_ = _aligned_malloc(size, alignof(std::max_align_t));
#else
        entry_ptr_ = std::aligned_alloc(alignof(std::max_align_t), size);
#endif
        if (!entry_ptr_) {
            throw std::bad_alloc();
        }
        
        // Zero-initialize
        std::memset(entry_ptr_, 0, size);
    }

    /**
     * @brief Destructor - automatic cleanup
     */
    ~RAIIHashEntry() {
        if (entry_ptr_) {
#ifdef _WIN32
            _aligned_free(entry_ptr_);
#else
            std::free(entry_ptr_);
#endif
        }
    }

    // Non-copyable, movable
    RAIIHashEntry(const RAIIHashEntry&) = delete;
    RAIIHashEntry& operator=(const RAIIHashEntry&) = delete;
    
    RAIIHashEntry(RAIIHashEntry&& other) noexcept 
        : entry_ptr_(other.entry_ptr_), entry_size_(other.entry_size_) {
        other.entry_ptr_ = nullptr;
        other.entry_size_ = 0;
    }

    RAIIHashEntry& operator=(RAIIHashEntry&& other) noexcept {
        if (this != &other) {
            if (entry_ptr_) {
#ifdef _WIN32
                _aligned_free(entry_ptr_);
#else
                std::free(entry_ptr_);
#endif
            }
            entry_ptr_ = other.entry_ptr_;
            entry_size_ = other.entry_size_;
            other.entry_ptr_ = nullptr;
            other.entry_size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get raw pointer (for C API compatibility)
     * @return Raw pointer to entry data
     */
    void* get() noexcept {
        return entry_ptr_;
    }

    /**
     * @brief Get const raw pointer
     * @return Const raw pointer to entry data
     */
    const void* get() const noexcept {
        return entry_ptr_;
    }

    /**
     * @brief Get entry size
     * @return Size of allocated entry in bytes
     */
    size_t size() const noexcept {
        return entry_size_;
    }

    /**
     * @brief Check if entry is valid
     * @return true if entry is allocated
     */
    bool valid() const noexcept {
        return entry_ptr_ != nullptr;
    }

    /**
     * @brief Cast to specific type
     * @tparam T Target type
     * @return Typed pointer to entry data
     */
    template<typename T>
    T* as() noexcept {
        return static_cast<T*>(entry_ptr_);
    }

    /**
     * @brief Cast to specific type (const version)
     * @tparam T Target type
     * @return Const typed pointer to entry data
     */
    template<typename T>
    const T* as() const noexcept {
        return static_cast<const T*>(entry_ptr_);
    }
};

/**
 * @brief Memory statistics and monitoring
 * 
 * Provides memory usage tracking and statistics for debugging
 * and performance monitoring.
 */
class MemoryMonitor {
private:
    static std::atomic<size_t> total_allocated_;
    static std::atomic<size_t> peak_allocated_;
    static std::atomic<size_t> allocation_count_;

public:
    /**
     * @brief Record memory allocation
     * @param size Size of allocation
     */
    static void recordAllocation(size_t size) noexcept {
        total_allocated_.fetch_add(size, std::memory_order_relaxed);
        allocation_count_.fetch_add(1, std::memory_order_relaxed);
        
        // Update peak if necessary
        size_t current = total_allocated_.load(std::memory_order_relaxed);
        size_t peak = peak_allocated_.load(std::memory_order_relaxed);
        while (current > peak && 
               !peak_allocated_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
            // Retry if another thread updated peak
        }
    }

    /**
     * @brief Record memory deallocation
     * @param size Size of deallocation
     */
    static void recordDeallocation(size_t size) noexcept {
        total_allocated_.fetch_sub(size, std::memory_order_relaxed);
    }

    /**
     * @brief Get current allocated memory
     * @return Current allocated memory in bytes
     */
    static size_t getCurrentAllocated() noexcept {
        return total_allocated_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get peak allocated memory
     * @return Peak allocated memory in bytes
     */
    static size_t getPeakAllocated() noexcept {
        return peak_allocated_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get total allocation count
     * @return Number of allocations performed
     */
    static size_t getAllocationCount() noexcept {
        return allocation_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset statistics
     */
    static void reset() noexcept {
        total_allocated_.store(0, std::memory_order_relaxed);
        peak_allocated_.store(0, std::memory_order_relaxed);
        allocation_count_.store(0, std::memory_order_relaxed);
    }
};

// Static member definitions moved to inline to avoid multiple definition
inline std::atomic<size_t> MemoryMonitor::total_allocated_{0};
inline std::atomic<size_t> MemoryMonitor::peak_allocated_{0};
inline std::atomic<size_t> MemoryMonitor::allocation_count_{0};

/**
 * @brief Convenience type aliases for common use cases
 * Note: TH_PARAM is defined in Kangaroo.h
 */
// using ThreadParamArray = RAIIArray<TH_PARAM>;  // Commented out to avoid redefinition
using ThreadHandleArray = RAIIArray<void*>; // THREAD_HANDLE is platform-specific

#endif // MODERN_MEMORY_MANAGER_H