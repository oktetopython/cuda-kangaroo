/*
* Unified CUDA Memory Management System
* Provides RAII-based memory management for GPU operations
*/

#ifndef CUDA_MEMORY_MANAGER_H
#define CUDA_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "CudaErrorHandler.h"

// Memory allocation types
enum class MemoryType {
    DEVICE,           // GPU global memory
    HOST_PINNED,      // CPU pinned memory
    HOST_MAPPED,      // CPU mapped memory
    UNIFIED           // Unified memory (CUDA 6.0+)
};

// Memory allocation flags
enum class MemoryFlags {
    DEFAULT = 0,
    WRITE_COMBINED = cudaHostAllocWriteCombined,
    MAPPED = cudaHostAllocMapped,
    PORTABLE = cudaHostAllocPortable
};

// RAII Memory Block
class CudaMemoryBlock {
private:
    void* ptr_;
    size_t size_;
    MemoryType type_;
    bool valid_;

public:
    CudaMemoryBlock(size_t size, MemoryType type, MemoryFlags flags = MemoryFlags::DEFAULT) 
        : ptr_(nullptr), size_(size), type_(type), valid_(false) {
        
        cudaError_t err = cudaSuccess;
        
        switch(type) {
            case MemoryType::DEVICE:
                err = cudaMalloc(&ptr_, size);
                break;
                
            case MemoryType::HOST_PINNED:
                err = cudaHostAlloc(&ptr_, size, static_cast<unsigned int>(flags));
                break;
                
            case MemoryType::HOST_MAPPED:
                err = cudaHostAlloc(&ptr_, size, cudaHostAllocMapped | static_cast<unsigned int>(flags));
                break;
                
            case MemoryType::UNIFIED:
                err = cudaMallocManaged(&ptr_, size);
                break;
        }
        
        if(err == cudaSuccess) {
            valid_ = true;
        } else {
            CudaErrorHandler::HandleMemoryError(err, GetTypeString(), size);
        }
    }
    
    ~CudaMemoryBlock() {
        if(valid_ && ptr_) {
            cudaError_t err = cudaSuccess;
            
            switch(type_) {
                case MemoryType::DEVICE:
                case MemoryType::UNIFIED:
                    err = cudaFree(ptr_);
                    break;
                    
                case MemoryType::HOST_PINNED:
                case MemoryType::HOST_MAPPED:
                    err = cudaFreeHost(ptr_);
                    break;
            }
            
            if(err != cudaSuccess) {
                printf("Warning: Failed to free %s memory: %s\n", 
                       GetTypeString(), cudaGetErrorString(err));
            }
        }
    }
    
    // Move constructor
    CudaMemoryBlock(CudaMemoryBlock&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_), type_(other.type_), valid_(other.valid_) {
        other.ptr_ = nullptr;
        other.valid_ = false;
    }
    
    // Move assignment
    CudaMemoryBlock& operator=(CudaMemoryBlock&& other) noexcept {
        if(this != &other) {
            // Clean up current resources
            this->~CudaMemoryBlock();
            
            // Move from other
            ptr_ = other.ptr_;
            size_ = other.size_;
            type_ = other.type_;
            valid_ = other.valid_;
            
            other.ptr_ = nullptr;
            other.valid_ = false;
        }
        return *this;
    }
    
    // Delete copy operations
    CudaMemoryBlock(const CudaMemoryBlock&) = delete;
    CudaMemoryBlock& operator=(const CudaMemoryBlock&) = delete;
    
    // Accessors
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool valid() const { return valid_; }
    MemoryType type() const { return type_; }
    
    // Type casting operators
    template<typename T>
    T* as() const { return static_cast<T*>(ptr_); }
    
    // Memory operations
    bool copyFrom(const void* src, size_t bytes, cudaMemcpyKind kind = cudaMemcpyDefault) {
        if(!valid_ || bytes > size_) return false;
        
        cudaError_t err = cudaMemcpy(ptr_, src, bytes, kind);
        if(err != cudaSuccess) {
            CudaErrorHandler::HandleGPUEngineError(err, "Memory copy");
            return false;
        }
        return true;
    }
    
    bool copyTo(void* dst, size_t bytes, cudaMemcpyKind kind = cudaMemcpyDefault) const {
        if(!valid_ || bytes > size_) return false;
        
        cudaError_t err = cudaMemcpy(dst, ptr_, bytes, kind);
        if(err != cudaSuccess) {
            CudaErrorHandler::HandleGPUEngineError(err, "Memory copy");
            return false;
        }
        return true;
    }
    
    bool zero() {
        if(!valid_) return false;
        
        cudaError_t err = cudaMemset(ptr_, 0, size_);
        if(err != cudaSuccess) {
            CudaErrorHandler::HandleGPUEngineError(err, "Memory zero");
            return false;
        }
        return true;
    }

private:
    const char* GetTypeString() const {
        switch(type_) {
            case MemoryType::DEVICE: return "device";
            case MemoryType::HOST_PINNED: return "host pinned";
            case MemoryType::HOST_MAPPED: return "host mapped";
            case MemoryType::UNIFIED: return "unified";
            default: return "unknown";
        }
    }
};

// Memory Pool for efficient allocation/deallocation
class CudaMemoryPool {
private:
    struct PoolBlock {
        std::unique_ptr<CudaMemoryBlock> memory;
        bool in_use;
        size_t size;
        
        PoolBlock(size_t s, MemoryType type) 
            : memory(std::make_unique<CudaMemoryBlock>(s, type)), in_use(false), size(s) {}
    };
    
    std::vector<PoolBlock> pool_;
    MemoryType pool_type_;
    
public:
    CudaMemoryPool(MemoryType type) : pool_type_(type) {}
    
    CudaMemoryBlock* allocate(size_t size) {
        // Try to find existing block
        for(auto& block : pool_) {
            if(!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.memory.get();
            }
        }
        
        // Create new block
        pool_.emplace_back(size, pool_type_);
        auto& new_block = pool_.back();
        if(new_block.memory->valid()) {
            new_block.in_use = true;
            return new_block.memory.get();
        }
        
        // Remove failed allocation
        pool_.pop_back();
        return nullptr;
    }
    
    void deallocate(CudaMemoryBlock* block) {
        for(auto& pool_block : pool_) {
            if(pool_block.memory.get() == block) {
                pool_block.in_use = false;
                break;
            }
        }
    }
    
    void clear() {
        pool_.clear();
    }
    
    size_t total_allocated() const {
        size_t total = 0;
        for(const auto& block : pool_) {
            total += block.size;
        }
        return total;
    }
};

// Kangaroo-specific memory layout
struct KangarooMemoryLayout {
    // Kangaroo data: x[4], y[4], distance[2], lastJump[1] per kangaroo
    static constexpr size_t KSIZE = 11;  // 64-bit words per kangaroo
    
    size_t kangaroo_count;
    size_t group_size;
    
    // Memory blocks
    std::unique_ptr<CudaMemoryBlock> device_kangaroos;
    std::unique_ptr<CudaMemoryBlock> pinned_kangaroos;
    std::unique_ptr<CudaMemoryBlock> device_output;
    std::unique_ptr<CudaMemoryBlock> pinned_output;
    std::unique_ptr<CudaMemoryBlock> jump_points;
    
    KangarooMemoryLayout(size_t kangaroo_count, size_t group_size, size_t max_found) 
        : kangaroo_count(kangaroo_count), group_size(group_size) {
        
        size_t kangaroo_size = kangaroo_count * group_size * KSIZE * sizeof(uint64_t);
        size_t pinned_size = group_size * KSIZE * sizeof(uint64_t);
        size_t output_size = (max_found * 14 + 1) * sizeof(uint32_t); // ITEM_SIZE32 = 14
        size_t jump_size = 256 * 4 * sizeof(uint64_t); // NB_JUMP * 4 coordinates
        
        // Allocate all memory blocks
        device_kangaroos = std::make_unique<CudaMemoryBlock>(kangaroo_size, MemoryType::DEVICE);
        pinned_kangaroos = std::make_unique<CudaMemoryBlock>(pinned_size, MemoryType::HOST_PINNED, 
                                                           MemoryFlags::WRITE_COMBINED);
        device_output = std::make_unique<CudaMemoryBlock>(output_size, MemoryType::DEVICE);
        pinned_output = std::make_unique<CudaMemoryBlock>(output_size, MemoryType::HOST_MAPPED);
        jump_points = std::make_unique<CudaMemoryBlock>(jump_size, MemoryType::HOST_PINNED);
    }
    
    bool all_valid() const {
        return device_kangaroos->valid() && 
               pinned_kangaroos->valid() && 
               device_output->valid() && 
               pinned_output->valid() && 
               jump_points->valid();
    }
    
    size_t total_memory() const {
        return device_kangaroos->size() + 
               pinned_kangaroos->size() + 
               device_output->size() + 
               pinned_output->size() + 
               jump_points->size();
    }
};

// Global memory manager instance
class CudaMemoryManager {
private:
    static std::unique_ptr<CudaMemoryPool> device_pool_;
    static std::unique_ptr<CudaMemoryPool> host_pool_;
    
public:
    static void Initialize() {
        if(!device_pool_) {
            device_pool_ = std::make_unique<CudaMemoryPool>(MemoryType::DEVICE);
        }
        if(!host_pool_) {
            host_pool_ = std::make_unique<CudaMemoryPool>(MemoryType::HOST_PINNED);
        }
    }
    
    static CudaMemoryBlock* AllocateDevice(size_t size) {
        Initialize();
        return device_pool_->allocate(size);
    }
    
    static CudaMemoryBlock* AllocateHost(size_t size) {
        Initialize();
        return host_pool_->allocate(size);
    }
    
    static void DeallocateDevice(CudaMemoryBlock* block) {
        if(device_pool_) device_pool_->deallocate(block);
    }
    
    static void DeallocateHost(CudaMemoryBlock* block) {
        if(host_pool_) host_pool_->deallocate(block);
    }
    
    static void Cleanup() {
        device_pool_.reset();
        host_pool_.reset();
    }
    
    static size_t GetTotalAllocated() {
        size_t total = 0;
        if(device_pool_) total += device_pool_->total_allocated();
        if(host_pool_) total += host_pool_->total_allocated();
        return total;
    }
};

// Static member definitions
std::unique_ptr<CudaMemoryPool> CudaMemoryManager::device_pool_ = nullptr;
std::unique_ptr<CudaMemoryPool> CudaMemoryManager::host_pool_ = nullptr;

#endif // CUDA_MEMORY_MANAGER_H