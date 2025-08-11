#!/bin/bash

# Bitcoin Puzzle 135 System - Linux Build Script
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "🚀 BITCOIN PUZZLE 135 SYSTEM - LINUX BUILD SCRIPT"
echo "===================================================="
echo -e "${NC}"
echo -e "${YELLOW}🔧 Building all components for Linux...${NC}"
echo

# Function to check command success
check_success() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ $1 failed!${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ $1 successful!${NC}"
    fi
}

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${BLUE}📁 Creating build directory...${NC}"
    mkdir build
fi

cd build

echo
echo -e "${PURPLE}🔨 Step 1: Configuring CMake...${NC}"
echo "================================"
cmake .. -DCMAKE_BUILD_TYPE=Release
check_success "CMake configuration"

echo
echo -e "${PURPLE}🔨 Step 2: Building Main Kangaroo Program...${NC}"
echo "============================================="
make kangaroo -j$(nproc)
check_success "Kangaroo build"

echo
echo -e "${PURPLE}🔨 Step 3: Building Bitcoin Puzzle 135 Challenge...${NC}"
echo "==================================================="
make puzzle135_challenge -j$(nproc)
check_success "Puzzle135 Challenge build"

echo
echo -e "${PURPLE}🔨 Step 4: Building Puzzle135 Table Generator...${NC}"
echo "================================================"
make puzzle135_bl_generator -j$(nproc)
check_success "Puzzle135 Generator build"

echo
echo -e "${PURPLE}🔨 Step 5: Building Real EC Table Generator...${NC}"
echo "=============================================="
make generate_bl_real_ec_table -j$(nproc)
check_success "Real EC Generator build"

echo
echo -e "${PURPLE}🔨 Step 6: Building System Tests...${NC}"
echo "==================================="
make test_puzzle135_system -j$(nproc)
check_success "System Test build"

make test_small_puzzle -j$(nproc)
check_success "Small Puzzle Test build"

echo
echo -e "${PURPLE}🔨 Step 7: Building Performance Benchmark...${NC}"
echo "============================================"
make performance_benchmark -j$(nproc)
check_success "Performance Benchmark build"

echo
echo -e "${GREEN}🎉 BUILD COMPLETE! All components built successfully!${NC}"
echo "====================================================="
echo
echo -e "${CYAN}📦 Built Programs:${NC}"
echo "├── kangaroo                   - Main GPU-accelerated solver"
echo "├── puzzle135_challenge        - Bitcoin Puzzle 135 challenge"
echo "├── puzzle135_bl_generator     - Puzzle 135 table generator"
echo "├── generate_bl_real_ec_table  - Real EC table generator"
echo "├── test_puzzle135_system      - System verification"
echo "├── test_small_puzzle          - Small-scale algorithm test"
echo "└── performance_benchmark      - Performance benchmark"
echo
echo -e "${YELLOW}🚀 Quick Start:${NC}"
echo "1. Run system test: ./Release/test_puzzle135_system"
echo "2. Generate table:  ./Release/puzzle135_bl_generator 40 8192 table.bin"
echo "3. Start challenge: ./Release/puzzle135_challenge table.bin 1000000000"
echo
echo -e "${BLUE}📚 For detailed usage, see COMPLETE_USER_GUIDE.md${NC}"
echo

cd ..

echo -e "${YELLOW}🔧 Creating convenience scripts...${NC}"
echo

# Create quick test script
cat > quick_test.sh << 'EOF'
#!/bin/bash
echo "🧪 Running Quick System Test..."
./build/Release/test_puzzle135_system
read -p "Press Enter to continue..."
EOF
chmod +x quick_test.sh

# Create table generation script
cat > generate_table.sh << 'EOF'
#!/bin/bash
echo "📊 Generating Precompute Table..."
if [ $# -ne 3 ]; then
    echo "Usage: $0 <L> <T> <output_file>"
    echo "Example: $0 40 8192 puzzle135_table.bin"
    echo
    echo "Parameters:"
    echo "  L: Search range size (2^L)"
    echo "  T: Table size"
    echo "  output_file: Output file path"
    exit 1
fi

echo "Generating table with L=$1, T=$2, output=$3"
./build/Release/puzzle135_bl_generator $1 $2 $3
echo "Table generation complete!"
read -p "Press Enter to continue..."
EOF
chmod +x generate_table.sh

# Create challenge script
cat > start_challenge.sh << 'EOF'
#!/bin/bash
echo "🎯 Starting Bitcoin Puzzle 135 Challenge..."
if [ $# -ne 2 ]; then
    echo "Usage: $0 <table_file> <max_steps>"
    echo "Example: $0 puzzle135_table.bin 1000000000"
    echo
    echo "Parameters:"
    echo "  table_file: Precompute table file"
    echo "  max_steps: Maximum search steps"
    exit 1
fi

echo "Starting challenge with table=$1, max_steps=$2"
./build/Release/puzzle135_challenge $1 $2
EOF
chmod +x start_challenge.sh

# Create monitoring script
cat > monitor_progress.sh << 'EOF'
#!/bin/bash
echo "📊 Bitcoin Puzzle 135 Progress Monitor"
echo "======================================"
echo "This script monitors the challenge progress in real-time"
echo

# Check if log file exists
if [ ! -f "puzzle135_progress.log" ]; then
    echo "⚠️  No progress log found. Start the challenge first."
    exit 1
fi

echo "📈 Monitoring progress... (Press Ctrl+C to stop)"
echo
tail -f puzzle135_progress.log
EOF
chmod +x monitor_progress.sh

# Create system info script
cat > system_info.sh << 'EOF'
#!/bin/bash
echo "🖥️  System Information for Bitcoin Puzzle 135"
echo "=============================================="
echo
echo "📋 Operating System:"
lsb_release -a 2>/dev/null || cat /etc/os-release
echo
echo "🔧 CPU Information:"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core"
echo
echo "💾 Memory Information:"
free -h
echo
echo "🎮 GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
else
    echo "NVIDIA GPU not detected or nvidia-smi not available"
fi
echo
echo "🔨 CUDA Information:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "CUDA not detected or nvcc not available"
fi
echo
echo "📦 CMake Version:"
cmake --version | head -1
echo
echo "🔧 GCC Version:"
gcc --version | head -1
EOF
chmod +x system_info.sh

echo -e "${GREEN}✅ Convenience scripts created:${NC}"
echo "├── quick_test.sh       - Quick system verification"
echo "├── generate_table.sh   - Generate precompute tables"
echo "├── start_challenge.sh  - Start Bitcoin Puzzle challenge"
echo "├── monitor_progress.sh - Monitor challenge progress"
echo "└── system_info.sh      - Display system information"
echo

echo -e "${GREEN}🎉 LINUX BUILD COMPLETE!${NC}"
echo "=========================="
echo "All programs and scripts are ready to use!"
echo
echo -e "${YELLOW}💡 Next Steps:${NC}"
echo "1. Run ./system_info.sh to check your system"
echo "2. Run ./quick_test.sh to verify installation"
echo "3. Use ./generate_table.sh to create precompute tables"
echo "4. Use ./start_challenge.sh to begin the challenge"
echo
