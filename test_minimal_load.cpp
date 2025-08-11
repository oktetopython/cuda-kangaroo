/**
 * Minimal test to load the real EC table
 */

#include "optimizations/phase4/bl_table_io_v2.h"
#include <iostream>
#include <fstream>

int main() {
    std::ofstream logfile("test_minimal_load_output.txt");
    logfile << "Testing table load..." << std::endl;
    std::cout << "Testing table load..." << std::endl;
    
    PrecomputeTableHeader header;
    std::vector<PrecomputeTableEntry> entries;
    
    std::string filename = "bl_real_ec_table_L20_T1024.bin";
    
    if (PrecomputeTableLoader::loadTable(filename, header, entries)) {
        logfile << "SUCCESS: Table loaded!" << std::endl;
        logfile << "Entries: " << entries.size() << std::endl;
        logfile << "L: " << header.L << std::endl;
        logfile << "T: " << header.T << std::endl;
        logfile << "W: " << header.W << std::endl;
        std::cout << "SUCCESS: Table loaded!" << std::endl;
        std::cout << "Entries: " << entries.size() << std::endl;
        std::cout << "L: " << header.L << std::endl;
        std::cout << "T: " << header.T << std::endl;
        std::cout << "W: " << header.W << std::endl;
    } else {
        logfile << "FAILED: Could not load table" << std::endl;
        std::cout << "FAILED: Could not load table" << std::endl;
        return 1;
    }

    logfile.close();
    
    return 0;
}
