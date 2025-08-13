/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef COMMONUTILSH
#define COMMONUTILSH

#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "SECPK1/Int.h"

// Structure for distinguished points
typedef struct
{
  Int x;
  Int d;
  uint64_t kIdx;
} ITEM;

// Common error handling functions
namespace CommonUtils {

  // Unified error reporting function
  bool reportError(const std::string& context, const std::string& message);
  
  // Unified parameter validation
  void printInvalidArgument(const std::string& name);
  
  // File operation helpers
  bool safeFileOpen(const std::string& fileName, const std::string& mode, FILE** file);
  void safeFileClose(FILE* file);
  
  // Progress display helpers
  void printProgress(const char* symbol = ".");
  void printProgressWithPercent(int current, int total);
  
  // String parsing helpers
  int getInt(const std::string& name, const char* value);
  double getDouble(const std::string& name, const char* value);
  void getInts(const std::string& name, std::vector<int>& tokens, const std::string& text, char sep);
  
  // Time formatting helpers
  std::string formatTime(double seconds);
  std::string formatTimeStr(double seconds);
  
  // Memory management helpers
  template<typename T>
  void safeDelete(T*& ptr) {
    if(ptr) {
      delete ptr;
      ptr = nullptr;
    }
  }
  
  template<typename T>
  void safeDeleteArray(T*& ptr) {
    if(ptr) {
      delete[] ptr;
      ptr = nullptr;
    }
  }
  
  // Thread management helpers
  struct ThreadParams {
    int threadId;
    bool isRunning;
    bool hasStarted;
    bool isWaiting;
    void* obj;
  };
  
  // Constants
  namespace Constants {
    const int DEFAULT_SERVER_PORT = 17403;
    const int DEFAULT_TIMEOUT_MS = 3000;
    const int DEFAULT_SAVE_PERIOD = 60;
    const int DEFAULT_THREAD_COUNT = 1;
  }
  
} // namespace CommonUtils

#endif // COMMONUTILSH
