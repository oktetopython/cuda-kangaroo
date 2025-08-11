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

echo -e "${BLUE}🚀 CUDA-BSGS-Kangaroo Linux Build Script${NC}"
echo "=================================================="

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}❌ This script is designed for Linux systems${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"

if ! command_exists cmake; then
    echo -e "${RED}❌ CMake not found. Please install cmake${NC}"
    echo "Ubuntu/Debian: sudo apt install cmake"
    echo "CentOS/RHEL: sudo yum install cmake"
    exit 1
fi

if ! command_exists make; then
    echo -e "${RED}❌ Make not found. Please install build-essential${NC}"
    echo "Ubuntu/Debian: sudo apt install build-essential"
    exit 1
fi

if ! command_exists g++; then
    echo -e "${RED}❌ G++ not found. Please install g++${NC}"
    echo "Ubuntu/Debian: sudo apt install g++"
    exit 1
fi

# Check for CUDA (optional)
CUDA_AVAILABLE=false
if command_exists nvcc; then
    CUDA_AVAILABLE=true
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo -e "${GREEN}✅ CUDA found: $(nvcc --version | grep release)${NC}"
    echo -e "${BLUE}📋 CUDA Version: $CUDA_VERSION${NC}"

    # Check CUDA version compatibility
    if [[ $(echo "$CUDA_VERSION < 11.0" | bc -l) -eq 1 ]]; then
        echo -e "${YELLOW}⚠️  CUDA version < 11.0 detected. Some GPU architectures will be disabled${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  CUDA not found. GPU features will be disabled${NC}"
fi

echo -e "${GREEN}✅ All required dependencies found${NC}"

# Create build directory
echo -e "${YELLOW}📁 Creating build directory...${NC}"
mkdir -p build_linux
cd build_linux

# Configure with CMake
echo -e "${YELLOW}⚙️  Configuring with CMake...${NC}"
if [ "$CUDA_AVAILABLE" = true ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release
else
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITHGPU=OFF
fi

# Build
echo -e "${YELLOW}🔨 Building project...${NC}"
make -j$(nproc) kangaroo

# Check if build was successful
if [ -f "kangaroo" ]; then
    echo -e "${GREEN}🎉 Build successful!${NC}"
    echo "=================================================="
    echo -e "${BLUE}📋 Build Summary:${NC}"
    echo "  • Executable: $(pwd)/kangaroo"
    echo "  • CUDA Support: $CUDA_AVAILABLE"
    echo "  • Threads used: $(nproc)"
    
    # Test the executable
    echo -e "${YELLOW}🧪 Testing executable...${NC}"
    if ./kangaroo 2>/dev/null | grep -q "Kangaroo"; then
        echo -e "${GREEN}✅ Executable test passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Executable test failed, but binary exists${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}🚀 Usage:${NC}"
    echo "  cd build_linux"
    echo "  ./kangaroo [options] input_file"
    echo ""
    echo -e "${BLUE}📖 For help:${NC}"
    echo "  ./kangaroo -h"
    
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
