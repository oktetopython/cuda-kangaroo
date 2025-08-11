#!/bin/bash

# Bitcoin Puzzle 135 Challenge Monitor - Linux Version
# Make executable with: chmod +x monitor_puzzle135.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear

echo -e "${CYAN}"
echo "🎯 Bitcoin Puzzle 135 Challenge Monitor"
echo "======================================"
echo -e "${NC}"
echo -e "${YELLOW}Target: 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v${NC}"
echo -e "${BLUE}Range: [4000000000000000000000000000000000, 7fffffffffffffffffffffffffffffffff]${NC}"
echo -e "${GREEN}Prize: ~32 BTC${NC}"
echo ""

while true; do
    echo ""
    echo -e "${CYAN}📊 Current Status [$(date)]:${NC}"
    echo "====================================="
    
    if [ -f "puzzle135_progress.log" ]; then
        echo -e "${GREEN}📈 Latest Progress:${NC}"
        tail -n 3 puzzle135_progress.log
        echo ""
    else
        echo -e "${YELLOW}⚠️  Progress log not found - challenge may not be running${NC}"
        echo ""
    fi
    
    if [ -f "puzzle135_performance.log" ]; then
        echo -e "${BLUE}🚀 Performance Stats:${NC}"
        tail -n 5 puzzle135_performance.log
        echo ""
    else
        echo -e "${YELLOW}⚠️  Performance log not found${NC}"
        echo ""
    fi
    
    if [ -f "puzzle135_checkpoint.dat" ]; then
        echo -e "${PURPLE}💾 Checkpoint Status:${NC}"
        ls -lh puzzle135_checkpoint.dat | awk '{print "File: " $9 ", Size: " $5 ", Modified: " $6 " " $7 " " $8}'
        echo ""
    else
        echo -e "${YELLOW}⚠️  No checkpoint file found${NC}"
        echo ""
    fi
    
    # Check if challenge process is running
    if pgrep -f "puzzle135_challenge" > /dev/null; then
        echo -e "${GREEN}✅ Challenge process is running${NC}"
        echo -e "${GREEN}PID: $(pgrep -f puzzle135_challenge)${NC}"
    else
        echo -e "${RED}❌ Challenge process not detected${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}🕐 Next update in 5 minutes...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    echo "====================================="
    
    # Wait 5 minutes (300 seconds) or until interrupted
    sleep 300
    
    # Clear screen for next update
    clear
    
    echo -e "${CYAN}"
    echo "🎯 Bitcoin Puzzle 135 Challenge Monitor"
    echo "======================================"
    echo -e "${NC}"
    echo -e "${YELLOW}Target: 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v${NC}"
    echo -e "${BLUE}Range: [4000000000000000000000000000000000, 7fffffffffffffffffffffffffffffffff]${NC}"
    echo -e "${GREEN}Prize: ~32 BTC${NC}"
done
