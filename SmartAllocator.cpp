#include "SmartAllocator.h"

// Static member definitions
std::vector<std::unique_ptr<SmartAllocator::MemoryBlock>> SmartAllocator::memory_blocks_;
std::mutex SmartAllocator::allocation_mutex_;
std::atomic<size_t> SmartAllocator::total_allocated_{0};
std::atomic<size_t> SmartAllocator::peak_allocated_{0};
