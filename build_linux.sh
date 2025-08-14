#!/bin/bash

# CUDA-BSGS-Kangaroo Linux Build Script
# This script builds the project on Linux systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}馃殌 CUDA-BSGS-Kangaroo Linux Build Script${NC}"
echo "=================================================="

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}鉂?This script is designed for Linux systems${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo -e "${YELLOW}馃攳 Checking dependencies...${NC}"

if ! command_exists cmake; then
    echo -e "${RED}鉂?CMake not found. Please install cmake${NC}"
    echo "Ubuntu/Debian: sudo apt install cmake"
    echo "CentOS/RHEL: sudo yum install cmake"
    exit 1
fi

if ! command_exists make; then
    echo -e "${RED}鉂?Make not found. Please install build-essential${NC}"
    echo "Ubuntu/Debian: sudo apt install build-essential"
    exit 1
fi

if ! command_exists g++; then
    echo -e "${RED}鉂?G++ not found. Please install g++${NC}"
    echo "Ubuntu/Debian: sudo apt install g++"
    exit 1
fi

# Check for CUDA (optional)
CUDA_AVAILABLE=false
if command_exists nvcc; then
    CUDA_AVAILABLE=true
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo -e "${GREEN}鉁?CUDA found: $(nvcc --version | grep release)${NC}"
    echo -e "${BLUE}馃搵 CUDA Version: $CUDA_VERSION${NC}"

    # Check CUDA version compatibility
    if [[ $(echo "$CUDA_VERSION < 11.0" | bc -l) -eq 1 ]]; then
        echo -e "${YELLOW}鈿狅笍  CUDA version < 11.0 detected. Some GPU architectures will be disabled${NC}"
    fi
else
    echo -e "${YELLOW}鈿狅笍  CUDA not found. GPU features will be disabled${NC}"
fi

echo -e "${GREEN}鉁?All required dependencies found${NC}"

# Create build directory
echo -e "${YELLOW}馃搧 Creating build directory...${NC}"
mkdir -p build_linux
cd build_linux

# Configure with CMake
echo -e "${YELLOW}鈿欙笍  Configuring with CMake...${NC}"
echo -e "${BLUE}🔍 Auto-detecting CUDA availability...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "${YELLOW}馃敤 Building project...${NC}"
make -j$(nproc) kangaroo

# Check if build was successful
if [ -f "kangaroo" ]; then
    echo -e "${GREEN}馃帀 Build successful!${NC}"
    echo "=================================================="
    echo -e "${BLUE}馃搵 Build Summary:${NC}"
    echo "  鈥?Executable: $(pwd)/kangaroo"
    echo "  鈥?CUDA Support: $CUDA_AVAILABLE"
    echo "  鈥?Threads used: $(nproc)"

    # Test the executable
    echo -e "${YELLOW}馃И Testing executable...${NC}"
    if ./kangaroo 2>/dev/null | grep -q "Kangaroo"; then
        echo -e "${GREEN}鉁?Executable test passed${NC}"
    else
        echo -e "${YELLOW}鈿狅笍  Executable test failed, but binary exists${NC}"
    fi

    echo ""
    echo -e "${BLUE}馃殌 Usage:${NC}"
    echo "  cd build_linux"
    echo "  ./kangaroo [options] input_file"
    echo ""
    echo -e "${BLUE}馃摉 For help:${NC}"
    echo "  ./kangaroo -h"

else
    echo -e "${RED}鉂?Build failed!${NC}"
    exit 1
fi
