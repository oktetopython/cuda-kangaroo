#include "UnifiedErrorHandler.h"

// Static member definitions
std::mutex UnifiedErrorHandler::log_mutex_;
std::ofstream UnifiedErrorHandler::log_file_;
ErrorLevel UnifiedErrorHandler::min_log_level_ = ErrorLevel::INFO;
std::atomic<bool> UnifiedErrorHandler::initialized_{false};
ErrorStats UnifiedErrorHandler::error_count_;
