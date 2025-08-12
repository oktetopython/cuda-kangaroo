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

#include "CommonUtils.h"
#include <stdexcept>
#include <sstream>
#include <cstring>

namespace CommonUtils {

// Unified error reporting function
bool reportError(const std::string& context, const std::string& message) {
  ::printf("Error: %s - %s\n", context.c_str(), message.c_str());
  return false;
}

// Unified parameter validation
void printInvalidArgument(const std::string& name) {
  printf("Invalid %s argument, number expected\n", name.c_str());
  exit(-1);
}

// File operation helpers
bool safeFileOpen(const std::string& fileName, const std::string& mode, FILE** file) {
  *file = fopen(fileName.c_str(), mode.c_str());
  if(*file == NULL) {
    reportError("File operation", "Cannot open " + fileName + " - " + strerror(errno));
    return false;
  }
  return true;
}

void safeFileClose(FILE* file) {
  if(file && file != stdout && file != stderr) {
    fclose(file);
  }
}

// Progress display helpers
void printProgress(const char* symbol) {
  ::printf("%s", symbol);
  fflush(stdout);
}

void printProgressWithPercent(int current, int total) {
  if(total > 0) {
    int percent = (current * 100) / total;
    ::printf("\rProgress: %d%% (%d/%d)", percent, current, total);
    fflush(stdout);
  }
}

// String parsing helpers
int getInt(const std::string& name, const char* value) {
  int result;
  try {
    result = std::stoi(std::string(value));
  } catch(std::invalid_argument&) {
    printInvalidArgument(name);
  }
  return result;
}

double getDouble(const std::string& name, const char* value) {
  double result;
  try {
    result = std::stod(std::string(value));
  } catch(std::invalid_argument&) {
    printInvalidArgument(name);
  }
  return result;
}

void getInts(const std::string& name, std::vector<int>& tokens, const std::string& text, char sep) {
  size_t start = 0, end = 0;
  tokens.clear();
  int item;
  
  try {
    while((end = text.find(sep, start)) != std::string::npos) {
      item = std::stoi(text.substr(start, end - start));
      tokens.push_back(item);
      start = end + 1;
    }
    
    item = std::stoi(text.substr(start));
    tokens.push_back(item);
    
  } catch(std::invalid_argument&) {
    printInvalidArgument(name);
  }
}

// Time formatting helpers
std::string formatTime(double seconds) {
  int hours = (int)(seconds / 3600.0);
  int minutes = (int)((seconds - hours * 3600.0) / 60.0);
  int secs = (int)(seconds - hours * 3600.0 - minutes * 60.0);
  
  char buffer[64];
  if(hours > 0) {
    snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", hours, minutes, secs);
  } else {
    snprintf(buffer, sizeof(buffer), "%02d:%02d", minutes, secs);
  }
  return std::string(buffer);
}

std::string formatTimeStr(double seconds) {
  return formatTime(seconds);
}

} // namespace CommonUtils
