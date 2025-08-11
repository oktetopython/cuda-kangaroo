/**
 * Simple test for the V2 online solver
 */

#include "optimizations/phase4/bl_online_solver_v2.h"
#include <iostream>

int main() {
    std::cout << "🧪 Simple V2 Solver Test" << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        BLOnlineDLPSolverV2 solver;
        
        std::cout << "✅ Solver created successfully" << std::endl;
        
        // Try to load the real EC table
        std::string table_file = "bl_real_ec_table_L20_T1024.bin";
        std::cout << "Loading table: " << table_file << std::endl;
        
        if (solver.loadPrecomputeTableV2(table_file)) {
            std::cout << "✅ Table loaded successfully" << std::endl;
            
            if (solver.buildHashIndex()) {
                std::cout << "✅ Hash index built successfully" << std::endl;
                
                solver.printTableStatistics();
                
                std::cout << "\n🎯 Testing simple DLP solve..." << std::endl;
                DLPSolutionV2 solution = solver.solveDLPSimplified(12345);
                solver.printSolution(solution);
                
            } else {
                std::cout << "❌ Failed to build hash index" << std::endl;
            }
        } else {
            std::cout << "❌ Failed to load table" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 Simple test completed!" << std::endl;
    return 0;
}
