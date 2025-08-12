#ifndef SMART_ALLOCATOR_H
#define SMART_ALLOCATOR_H

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <vector>
#include <memory>
#include <stdexcept>

/**
 * Smart Memory Allocator with leak detection and automatic cleanup
 * Provides thread-safe memory management with debugging capabilities
 */
class SmartAllocator {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        std::atomic<bool> in_use;
        
        MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
        
        ~MemoryBlock() {
            if (ptr) {
                std::free(ptr);
                ptr = nullptr;
            }
        }
        
        // Disable copy
        MemoryBlock(const MemoryBlock&) = delete;
        MemoryBlock& operator=(const MemoryBlock&) = delete;
        
        // Enable move
        MemoryBlock(MemoryBlock&& other) noexcept 
            : ptr(other.ptr), size(other.size), in_use(other.in_use.load()) {
            other.ptr = nullptr;
            other.size = 0;
            other.in_use = false;
        }
    };
    
    static std::vector<std::unique_ptr<MemoryBlock>> memory_blocks_;
    static std::mutex allocation_mutex_;
    static std::atomic<size_t> total_allocated_;
    static std::atomic<size_t> peak_allocated_;
    
public:
    static void* allocate(size_t size) {
        if (size == 0) return nullptr;

        void* ptr = std::malloc(size);
        if (!ptr) {
            throw std::runtime_error("Memory allocation failed");
        }

        std::lock_guard<std::mutex> lock(allocation_mutex_);
        memory_blocks_.emplace_back(std::make_unique<MemoryBlock>(ptr, size));

        // Mark the memory block as in use
        memory_blocks_.back()->in_use = true;

        size_t current = total_allocated_.fetch_add(size) + size;
        size_t peak = peak_allocated_.load();
        while (current > peak && !peak_allocated_.compare_exchange_weak(peak, current)) {
            peak = peak_allocated_.load();
        }

        return ptr;
    }
    
    static void deallocate(void* ptr) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(allocation_mutex_);
        for (auto it = memory_blocks_.begin(); it != memory_blocks_.end(); ++it) {
            if ((*it)->ptr == ptr) {
                // Mark as not in use before deallocation
                (*it)->in_use = false;
                total_allocated_.fetch_sub((*it)->size);
                memory_blocks_.erase(it);
                std::free(ptr);
                return;
            }
        }
        // If not found in our tracking, still free it
        std::free(ptr);
    }
    
    static size_t getTotalAllocated() {
        return total_allocated_.load();
    }
    
    static size_t getPeakAllocated() {
        return peak_allocated_.load();
    }
    
    static void cleanup() {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        // Mark all blocks as not in use before cleanup
        for (auto& block : memory_blocks_) {
            if (block) {
                block->in_use = false;
            }
        }
        memory_blocks_.clear();
        total_allocated_ = 0;
    }

    // Get count of memory blocks currently in use
    static size_t getInUseCount() {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        size_t count = 0;
        for (const auto& block : memory_blocks_) {
            if (block && block->in_use.load()) {
                count++;
            }
        }
        return count;
    }

    // Get total size of memory blocks currently in use
    static size_t getInUseSize() {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        size_t size = 0;
        for (const auto& block : memory_blocks_) {
            if (block && block->in_use.load()) {
                size += block->size;
            }
        }
        return size;
    }

    // Check if a specific pointer is currently in use
    static bool isInUse(void* ptr) {
        if (!ptr) return false;

        std::lock_guard<std::mutex> lock(allocation_mutex_);
        for (const auto& block : memory_blocks_) {
            if (block && block->ptr == ptr) {
                return block->in_use.load();
            }
        }
        return false;
    }
};

#endif // SMART_ALLOCATOR_H
