#!/bin/bash

# Quick GPU Optimization for CUDA-BSGS-Kangaroo
# Focus on immediate, practical performance gains

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Quick GPU Optimization for CUDA-BSGS-Kangaroo${NC}"
echo "=================================================="

# Detect GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✅ GPU: $GPU_NAME${NC}"
    echo -e "${GREEN}✅ Memory: ${GPU_MEMORY}MB${NC}"
else
    echo -e "${RED}❌ No GPU detected${NC}"
    exit 1
fi

# Test different grid configurations
echo -e "${YELLOW}🧪 Testing GPU grid configurations...${NC}"

# Create test input for 135 puzzle
cat > test_input.txt << 'EOF'
4000000000000000000000000000000000
7fffffffffffffffffffffffffffffffff
02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
EOF

# Test configurations based on GPU
case "$GPU_NAME" in
    *"Tesla T4"*)
        GRID_CONFIGS=("272,256" "340,320" "408,384")
        ;;
    *"RTX 2080"*|*"RTX 3080"*|*"RTX 4090"*)
        GRID_CONFIGS=("272,256" "408,384" "544,512" "680,512")
        ;;
    *)
        GRID_CONFIGS=("272,256" "340,320")
        ;;
esac

BEST_PERF=0
BEST_CONFIG="272,256"

for config in "${GRID_CONFIGS[@]}"; do
    echo -e "${BLUE}Testing grid: $config${NC}"
    
    # Run 30-second test
    timeout 30s ./build_test/Release/kangaroo.exe -t 1 -gpu -g $config -d 22 test_input.txt > test_${config//,/_}.log 2>&1 || true
    
    # Extract performance
    PERF=$(grep -o 'GPU [0-9.]*MK/s' test_${config//,/_}.log | sed 's/GPU //;s/MK\/s//' || echo "0")
    
    if (( $(echo "$PERF > $BEST_PERF" | bc -l) )); then
        BEST_PERF=$PERF
        BEST_CONFIG=$config
    fi
    
    echo -e "${GREEN}Grid $config: $PERF MK/s${NC}"
    rm -f test_${config//,/_}.log
done

echo -e "${GREEN}✅ Best configuration: Grid($BEST_CONFIG) = $BEST_PERF MK/s${NC}"

# GPU performance tuning
echo -e "${YELLOW}🔧 Applying GPU performance settings...${NC}"

if [ "$EUID" -eq 0 ]; then
    # Set persistence mode
    nvidia-smi -pm 1
    
    # Set maximum clocks
    nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.graphics --format=csv,noheader,nounits | tr ',' ' ')
    
    echo -e "${GREEN}✅ GPU performance settings applied${NC}"
else
    echo -e "${YELLOW}⚠️  Run as root for GPU tuning: sudo $0${NC}"
fi

# Create optimized run script
cat > run_optimized.sh << EOF
#!/bin/bash

# Optimized run script for CUDA-BSGS-Kangaroo
echo "🚀 Running with optimized settings..."
echo "GPU: $GPU_NAME"
echo "Grid: $BEST_CONFIG"
echo "Expected performance: $BEST_PERF MK/s"
echo ""

# Run with optimal settings
nice -n -10 ./build_test/Release/kangaroo.exe -t 8 -gpu -g $BEST_CONFIG -d 22 "\$@"
EOF

chmod +x run_optimized.sh

# Create monitoring script
cat > monitor_performance.sh << 'EOF'
#!/bin/bash

# Performance monitoring script
echo "📊 Performance Monitor (Ctrl+C to stop)"
echo "========================================"

while true; do
    # GPU status
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    
    # Clear screen and show status
    clear
    echo "📊 Real-time Performance Monitor"
    echo "================================"
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    echo "🎯 GPU Status:"
    echo "   Utilization: ${GPU_UTIL}%"
    echo "   Memory: ${GPU_MEM}%"
    echo "   Temperature: ${GPU_TEMP}°C"
    echo "   Power: ${GPU_POWER}W"
    echo ""
    echo "💡 Tips:"
    echo "   • Target GPU utilization: >95%"
    echo "   • Keep temperature: <80°C"
    echo "   • Monitor for 'Dead 0' in kangaroo output"
    
    sleep 2
done
EOF

chmod +x monitor_performance.sh

# Cleanup
rm -f test_input.txt

echo ""
echo -e "${BLUE}📋 Quick Optimization Complete!${NC}"
echo "================================"
echo ""
echo -e "${GREEN}✅ Optimal GPU Grid: $BEST_CONFIG${NC}"
echo -e "${GREEN}✅ Expected Performance: $BEST_PERF MK/s${NC}"
echo ""
echo -e "${BLUE}🚀 Usage:${NC}"
echo "• Run optimized: ./run_optimized.sh puzzle135.txt"
echo "• Monitor performance: ./monitor_performance.sh"
echo ""
echo -e "${BLUE}📊 Manual command:${NC}"
echo "./build_test/Release/kangaroo.exe -t 8 -gpu -g $BEST_CONFIG -d 22 puzzle135.txt"

echo -e "${GREEN}🎉 Ready for optimized Bitcoin puzzle solving!${NC}"
