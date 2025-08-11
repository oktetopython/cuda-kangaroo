#!/bin/bash

# Bitcoin Puzzle 135 Challenge - Linux Launch Script
# Make executable with: chmod +x start_puzzle135_challenge.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "🚀 BITCOIN PUZZLE 135 CHALLENGE - STARTING NOW!"
echo "================================================"
echo -e "${NC}"
echo -e "${YELLOW}🏆 HISTORIC ACHIEVEMENT LAUNCH${NC}"
echo "================================================"
echo ""
echo "This launches the complete implementation"
echo "of the Bernstein-Lange algorithm for Bitcoin Puzzle 135."
echo ""
echo -e "${CYAN}🎯 Target: Bitcoin Puzzle 135${NC}"
echo -e "${GREEN}💰 Prize: ~32 BTC${NC}"
echo -e "${BLUE}🔑 Address: 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v${NC}"
echo -e "${PURPLE}📊 Range: [4000000000000000000000000000000000, 7fffffffffffffffffffffffffffffffff]${NC}"
echo -e "${YELLOW}🎯 Algorithm: Bernstein-Lange with real secp256k1 operations${NC}"
echo -e "${GREEN}🎯 Performance: 240,000+ steps/second expected${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: This is a REAL Bitcoin challenge!${NC}"
echo -e "${RED}⚠️  If solution is found, secure the private key immediately!${NC}"
echo ""

read -p "Press Enter to continue..."

echo ""
echo -e "${YELLOW}🔥 Launching Bitcoin Puzzle 135 Challenge...${NC}"
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}❌ Build directory not found. Please run cmake and make first.${NC}"
    exit 1
fi

# Check if executable exists
if [ ! -f "build/Release/puzzle135_challenge" ] && [ ! -f "build/puzzle135_challenge" ]; then
    echo -e "${RED}❌ puzzle135_challenge executable not found.${NC}"
    echo "Please build the project first:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  make -j\$(nproc)"
    exit 1
fi

# Check if precompute table exists
if [ ! -f "puzzle135_bl_table.bin" ]; then
    echo -e "${YELLOW}⚠️  Precompute table not found. Generating...${NC}"
    
    if [ -f "build/Release/puzzle135_bl_generator" ]; then
        ./build/Release/puzzle135_bl_generator
    elif [ -f "build/puzzle135_bl_generator" ]; then
        ./build/puzzle135_bl_generator
    else
        echo -e "${RED}❌ puzzle135_bl_generator not found.${NC}"
        exit 1
    fi
fi

# Launch the challenge
if [ -f "build/Release/puzzle135_challenge" ]; then
    ./build/Release/puzzle135_challenge puzzle135_bl_table.bin 1000000000
elif [ -f "build/puzzle135_challenge" ]; then
    ./build/puzzle135_challenge puzzle135_bl_table.bin 1000000000
else
    echo -e "${RED}❌ puzzle135_challenge executable not found.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🏁 Challenge session ended.${NC}"
echo "Check logs for results and progress."
echo ""
read -p "Press Enter to exit..."
