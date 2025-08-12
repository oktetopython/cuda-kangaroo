#ifndef UNIFIED_ERROR_HANDLER_H
#define UNIFIED_ERROR_HANDLER_H

#include <string>
#include <fstream>
#include <mutex>
#include <atomic>
#include <ctime>
#include <iostream>
#include <stdexcept>

/**
 * Unified Error Handler for CUDA-BSGS-Kangaroo Project
 * Provides centralized error logging, debugging, and statistics
 */

enum class ErrorLevel {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    CRITICAL = 3,
    FATAL = 4
};

enum class ErrorType {
    MEMORY = 0,
    GPU = 1,
    HASH_TABLE = 2,
    NETWORK = 3,
    FILE_IO = 4,
    COMPUTATION = 5,
    THREAD_SAFETY = 6
};

struct ErrorInfo {
    ErrorLevel level;
    ErrorType type;
    std::string message;
    std::string file;
    int line;
    std::time_t timestamp;
    
    ErrorInfo(ErrorLevel l, ErrorType t, const std::string& msg, const std::string& f, int ln)
        : level(l), type(t), message(msg), file(f), line(ln) {
        timestamp = std::time(nullptr);
    }
};

struct ErrorStats {
    // Error level counters
    std::atomic<uint64_t> info_count{0};
    std::atomic<uint64_t> warning_count{0};
    std::atomic<uint64_t> error_count{0};
    std::atomic<uint64_t> critical_count{0};
    std::atomic<uint64_t> fatal_count{0};

    // Error type counters - all 7 categories
    std::atomic<uint64_t> memory_errors{0};
    std::atomic<uint64_t> gpu_errors{0};
    std::atomic<uint64_t> hash_table_errors{0};
    std::atomic<uint64_t> network_errors{0};
    std::atomic<uint64_t> file_io_errors{0};
    std::atomic<uint64_t> computation_errors{0};
    std::atomic<uint64_t> thread_safety_errors{0};

    // Default constructor
    ErrorStats() = default;

    // Copy constructor (copies values, not atomic objects)
    ErrorStats(const ErrorStats& other)
        : info_count(other.info_count.load())
        , warning_count(other.warning_count.load())
        , error_count(other.error_count.load())
        , critical_count(other.critical_count.load())
        , fatal_count(other.fatal_count.load())
        , memory_errors(other.memory_errors.load())
        , gpu_errors(other.gpu_errors.load())
        , hash_table_errors(other.hash_table_errors.load())
        , network_errors(other.network_errors.load())
        , file_io_errors(other.file_io_errors.load())
        , computation_errors(other.computation_errors.load())
        , thread_safety_errors(other.thread_safety_errors.load()) {}

    // Assignment operator
    ErrorStats& operator=(const ErrorStats& other) {
        if (this != &other) {
            info_count = other.info_count.load();
            warning_count = other.warning_count.load();
            error_count = other.error_count.load();
            critical_count = other.critical_count.load();
            fatal_count = other.fatal_count.load();
            memory_errors = other.memory_errors.load();
            gpu_errors = other.gpu_errors.load();
            hash_table_errors = other.hash_table_errors.load();
            network_errors = other.network_errors.load();
            file_io_errors = other.file_io_errors.load();
            computation_errors = other.computation_errors.load();
            thread_safety_errors = other.thread_safety_errors.load();
        }
        return *this;
    }
};

class UnifiedErrorHandler {
private:
    static std::mutex log_mutex_;
    static std::ofstream log_file_;
    static ErrorLevel min_log_level_;
    static std::atomic<bool> initialized_;
    static ErrorStats error_count_;
    
    static std::string GetLevelString(ErrorLevel level) {
        switch(level) {
            case ErrorLevel::INFO: return "INFO";
            case ErrorLevel::WARNING: return "WARNING";
            case ErrorLevel::ERROR: return "ERROR";
            case ErrorLevel::CRITICAL: return "CRITICAL";
            case ErrorLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
    
    static std::string GetTypeString(ErrorType type) {
        switch(type) {
            case ErrorType::MEMORY: return "MEMORY";
            case ErrorType::GPU: return "GPU";
            case ErrorType::HASH_TABLE: return "HASH_TABLE";
            case ErrorType::NETWORK: return "NETWORK";
            case ErrorType::FILE_IO: return "FILE_IO";
            case ErrorType::COMPUTATION: return "COMPUTATION";
            case ErrorType::THREAD_SAFETY: return "THREAD_SAFETY";
            default: return "UNKNOWN";
        }
    }
    
    static std::string FormatTimestamp(std::time_t timestamp) {
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&timestamp));
        return std::string(buffer);
    }
    
public:
    static void Initialize(const std::string& log_file_path = "kangaroo_errors.log", 
                          ErrorLevel min_level = ErrorLevel::INFO) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        if (!initialized_.load()) {
            min_log_level_ = min_level;
            log_file_.open(log_file_path, std::ios::app);
            if (log_file_.is_open()) {
                log_file_ << "\n=== Kangaroo Error Handler Initialized at " 
                         << FormatTimestamp(std::time(nullptr)) << " ===\n";
                log_file_.flush();
            }
            initialized_ = true;
        }
    }
    
    static void LogError(ErrorLevel level, ErrorType type, const std::string& message,
                        const std::string& file = "", int line = 0) {
        if (!initialized_.load()) {
            Initialize();
        }
        
        if (level < min_log_level_) return;
        
        // Update statistics
        switch(level) {
            case ErrorLevel::INFO: error_count_.info_count++; break;
            case ErrorLevel::WARNING: error_count_.warning_count++; break;
            case ErrorLevel::ERROR: error_count_.error_count++; break;
            case ErrorLevel::CRITICAL: error_count_.critical_count++; break;
            case ErrorLevel::FATAL: error_count_.fatal_count++; break;
        }
        
        // Update error type statistics - all 7 categories
        switch(type) {
            case ErrorType::MEMORY: error_count_.memory_errors++; break;
            case ErrorType::GPU: error_count_.gpu_errors++; break;
            case ErrorType::HASH_TABLE: error_count_.hash_table_errors++; break;
            case ErrorType::NETWORK: error_count_.network_errors++; break;
            case ErrorType::FILE_IO: error_count_.file_io_errors++; break;
            case ErrorType::COMPUTATION: error_count_.computation_errors++; break;
            case ErrorType::THREAD_SAFETY: error_count_.thread_safety_errors++; break;
        }
        
        std::string formatted_message = "[" + GetLevelString(level) + "][" + GetTypeString(type) + "] " + message;
        if (!file.empty()) {
            formatted_message += " (" + file + ":" + std::to_string(line) + ")";
        }
        
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        // Console output for critical errors
        if (level >= ErrorLevel::CRITICAL) {
            std::cerr << formatted_message << std::endl;
        }
        
        // File logging
        if (log_file_.is_open()) {
            log_file_ << FormatTimestamp(std::time(nullptr)) << " " << formatted_message << std::endl;
            log_file_.flush();
        }
        
        // Fatal errors cause program termination
        if (level == ErrorLevel::FATAL) {
            Cleanup();
            throw std::runtime_error("Fatal error: " + message);
        }
    }
    
    static ErrorStats GetErrorStats() {
        ErrorStats stats;
        // Copy all error level counters
        stats.info_count = error_count_.info_count.load();
        stats.warning_count = error_count_.warning_count.load();
        stats.error_count = error_count_.error_count.load();
        stats.critical_count = error_count_.critical_count.load();
        stats.fatal_count = error_count_.fatal_count.load();

        // Copy all error type counters - all 7 categories
        stats.memory_errors = error_count_.memory_errors.load();
        stats.gpu_errors = error_count_.gpu_errors.load();
        stats.hash_table_errors = error_count_.hash_table_errors.load();
        stats.network_errors = error_count_.network_errors.load();
        stats.file_io_errors = error_count_.file_io_errors.load();
        stats.computation_errors = error_count_.computation_errors.load();
        stats.thread_safety_errors = error_count_.thread_safety_errors.load();

        return stats;
    }
    
    static void Cleanup() {
        std::lock_guard<std::mutex> lock(log_mutex_);
        if (log_file_.is_open()) {
            log_file_ << "=== Error Handler Cleanup at " 
                     << FormatTimestamp(std::time(nullptr)) << " ===\n";
            log_file_.close();
        }
        initialized_ = false;
    }
};

// Convenience macros
#define LOG_INFO(type, msg) UnifiedErrorHandler::LogError(ErrorLevel::INFO, type, msg, __FILE__, __LINE__)
#define LOG_WARNING(type, msg) UnifiedErrorHandler::LogError(ErrorLevel::WARNING, type, msg, __FILE__, __LINE__)
#define LOG_ERROR(type, msg) UnifiedErrorHandler::LogError(ErrorLevel::ERROR, type, msg, __FILE__, __LINE__)
#define LOG_CRITICAL(type, msg) UnifiedErrorHandler::LogError(ErrorLevel::CRITICAL, type, msg, __FILE__, __LINE__)
#define LOG_FATAL(type, msg) UnifiedErrorHandler::LogError(ErrorLevel::FATAL, type, msg, __FILE__, __LINE__)

#define LOG_MEMORY_ERROR(operation, size) LOG_ERROR(ErrorType::MEMORY, std::string(operation) + " failed for " + std::to_string(size) + " bytes")
#define LOG_GPU_ERROR(msg) LOG_ERROR(ErrorType::GPU, msg)
#define LOG_HASH_ERROR(operation, hash) LOG_ERROR(ErrorType::HASH_TABLE, std::string(operation) + " failed for hash " + std::to_string(hash))

#endif // UNIFIED_ERROR_HANDLER_H
