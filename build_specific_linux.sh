#!/bin/bash

# Bitcoin Puzzle 135 System - Specific Component Builder (Linux)
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

show_usage() {
    echo
    echo -e "${CYAN}📖 USAGE: $0 [component]${NC}"
    echo
    echo -e "${YELLOW}🎯 Available Components:${NC}"
    echo "  kangaroo    - Main GPU-accelerated Kangaroo solver"
    echo "  puzzle135   - Bitcoin Puzzle 135 challenge components"
    echo "  generator   - Precompute table generators"
    echo "  tests       - System verification and test programs"
    echo "  benchmark   - Performance benchmark program"
    echo "  all         - Build all components"
    echo
    echo -e "${BLUE}💡 Examples:${NC}"
    echo "  $0 kangaroo"
    echo "  $0 puzzle135"
    echo "  $0 tests"
    echo
}

# Function to check command success
check_success() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ $1 failed!${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ $1 successful!${NC}"
    fi
}

echo -e "${CYAN}"
echo "🔨 BITCOIN PUZZLE 135 - SPECIFIC COMPONENT BUILDER"
echo "=================================================="
echo -e "${NC}"

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${BLUE}📁 Creating build directory...${NC}"
    mkdir build
fi

cd build

# Configure CMake if not already done
if [ ! -f "CMakeCache.txt" ]; then
    echo -e "${YELLOW}🔧 Configuring CMake...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    check_success "CMake configuration"
fi

# Build specific component based on parameter
case "$1" in
    "kangaroo")
        echo -e "${PURPLE}🚀 Building Main Kangaroo Program...${NC}"
        echo "==================================="
        make kangaroo -j$(nproc)
        check_success "Kangaroo build"
        echo -e "${GREEN}📍 Location: ./build/Release/kangaroo${NC}"
        ;;
    
    "puzzle135")
        echo -e "${PURPLE}🎯 Building Bitcoin Puzzle 135 Components...${NC}"
        echo "============================================"
        echo -e "${YELLOW}🔨 Building Puzzle135 Challenge...${NC}"
        make puzzle135_challenge -j$(nproc)
        check_success "Puzzle135 Challenge build"
        
        echo -e "${YELLOW}🔨 Building Puzzle135 Generator...${NC}"
        make puzzle135_bl_generator -j$(nproc)
        check_success "Puzzle135 Generator build"
        
        echo -e "${GREEN}📍 Locations:${NC}"
        echo "  ├── ./build/Release/puzzle135_challenge"
        echo "  └── ./build/Release/puzzle135_bl_generator"
        ;;
    
    "generator")
        echo -e "${PURPLE}📊 Building Table Generators...${NC}"
        echo "==============================="
        echo -e "${YELLOW}🔨 Building Puzzle135 Generator...${NC}"
        make puzzle135_bl_generator -j$(nproc)
        check_success "Puzzle135 Generator build"
        
        echo -e "${YELLOW}🔨 Building Real EC Generator...${NC}"
        make generate_bl_real_ec_table -j$(nproc)
        check_success "Real EC Generator build"
        
        echo -e "${GREEN}📍 Locations:${NC}"
        echo "  ├── ./build/Release/puzzle135_bl_generator"
        echo "  └── ./build/Release/generate_bl_real_ec_table"
        ;;
    
    "tests")
        echo -e "${PURPLE}🧪 Building Test Programs...${NC}"
        echo "============================"
        echo -e "${YELLOW}🔨 Building System Test...${NC}"
        make test_puzzle135_system -j$(nproc)
        check_success "System Test build"
        
        echo -e "${YELLOW}🔨 Building Small Puzzle Test...${NC}"
        make test_small_puzzle -j$(nproc)
        check_success "Small Puzzle Test build"
        
        echo -e "${GREEN}📍 Locations:${NC}"
        echo "  ├── ./build/Release/test_puzzle135_system"
        echo "  └── ./build/Release/test_small_puzzle"
        ;;
    
    "benchmark")
        echo -e "${PURPLE}📈 Building Performance Benchmark...${NC}"
        echo "===================================="
        make performance_benchmark -j$(nproc)
        check_success "Performance Benchmark build"
        echo -e "${GREEN}📍 Location: ./build/Release/performance_benchmark${NC}"
        ;;
    
    "all")
        echo -e "${PURPLE}🏗️ Building All Components...${NC}"
        echo "============================="
        cd ..
        ./build_all_linux.sh
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ Invalid option: $1${NC}"
        echo
        echo -e "${YELLOW}🎯 Valid options: kangaroo, puzzle135, generator, tests, benchmark, all${NC}"
        echo
        exit 1
        ;;
esac

cd ..

echo
echo -e "${GREEN}🎉 Build Complete!${NC}"
echo "================="
echo
echo -e "${YELLOW}🚀 Quick Commands:${NC}"
echo "  Test system:    ./build/Release/test_puzzle135_system"
echo "  Generate table: ./build/Release/puzzle135_bl_generator 40 8192 table.bin"
echo "  Start challenge:./build/Release/puzzle135_challenge table.bin 1000000000"
echo
echo -e "${BLUE}📚 For detailed usage, see COMPLETE_USER_GUIDE.md${NC}"
echo
