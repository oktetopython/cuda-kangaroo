/**
 * Debug Point Structure and Hash Calculation
 * 
 * This program investigates why all points have hash=0x0 in our DP detection test.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <chrono>
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"

class PointStructureDebug {
private:
    Secp256K1 secp;
    std::mt19937_64 rng;
    
public:
    PointStructureDebug() {
        secp.Init();
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rng.seed(seed);
        std::cout << "? Point Structure Debug initialized" << std::endl;
    }
    
    void debugPointStructure() {
        std::cout << "\n=== Point Structure Debug ===" << std::endl;
        
        // Test with known private key
        Int private_key;
        private_key.SetInt32(12345);
        Point point = secp.ComputePublicKey(&private_key);
        
        std::cout << "Private key: 12345" << std::endl;
        std::cout << "Point structure analysis:" << std::endl;
        
        // Check all bits64 elements
        std::cout << "  point.x.bits64[0]: 0x" << std::hex << point.x.bits64[0] << std::endl;
        std::cout << "  point.x.bits64[1]: 0x" << std::hex << point.x.bits64[1] << std::endl;
        std::cout << "  point.x.bits64[2]: 0x" << std::hex << point.x.bits64[2] << std::endl;
        std::cout << "  point.x.bits64[3]: 0x" << std::hex << point.x.bits64[3] << std::endl;
        
        std::cout << "  point.y.bits64[0]: 0x" << std::hex << point.y.bits64[0] << std::endl;
        std::cout << "  point.y.bits64[1]: 0x" << std::hex << point.y.bits64[1] << std::endl;
        std::cout << "  point.y.bits64[2]: 0x" << std::hex << point.y.bits64[2] << std::endl;
        std::cout << "  point.y.bits64[3]: 0x" << std::hex << point.y.bits64[3] << std::endl;
        
        // Check if point is valid
        std::cout << "Point validation:" << std::endl;
        bool x_nonzero = (point.x.bits64[0] != 0 || point.x.bits64[1] != 0 || 
                         point.x.bits64[2] != 0 || point.x.bits64[3] != 0);
        bool y_nonzero = (point.y.bits64[0] != 0 || point.y.bits64[1] != 0 || 
                         point.y.bits64[2] != 0 || point.y.bits64[3] != 0);
        
        std::cout << "  X coordinate non-zero: " << (x_nonzero ? "YES" : "NO") << std::endl;
        std::cout << "  Y coordinate non-zero: " << (y_nonzero ? "YES" : "NO") << std::endl;
    }
    
    void debugRandomGeneration() {
        std::cout << "\n=== Random Generation Debug ===" << std::endl;
        
        std::cout << "Testing 5 random private keys:" << std::endl;
        
        for (int i = 0; i < 5; i++) {
            Int random_key;
            random_key.Rand(32);  // 32-bit random number
            
            std::cout << "Test " << (i+1) << ":" << std::endl;
            std::cout << "  Random key: " << random_key.GetBase16() << std::endl;
            
            Point point = secp.ComputePublicKey(&random_key);
            
            std::cout << "  Point X[0]: 0x" << std::hex << point.x.bits64[0] << std::endl;
            std::cout << "  Point Y[0]: 0x" << std::hex << point.y.bits64[0] << std::endl;
            
            // Check if this point would be detected as DP with mask 0x7
            uint64_t hash = point.x.bits64[0];
            bool is_dp = ((hash & 0x7) == 0);
            std::cout << "  DP (mask 0x7): " << (is_dp ? "YES" : "NO") << std::endl;
            std::cout << std::endl;
        }
    }
    
    void debugHashCalculation() {
        std::cout << "\n=== Hash Calculation Debug ===" << std::endl;
        
        // Test different hash calculation methods
        Int test_key;
        test_key.SetInt32(99999);
        Point point = secp.ComputePublicKey(&test_key);
        
        std::cout << "Test key: 99999" << std::endl;
        std::cout << "Point coordinates:" << std::endl;
        std::cout << "  Full X: 0x" << std::hex << point.x.bits64[3] << point.x.bits64[2] 
                  << point.x.bits64[1] << point.x.bits64[0] << std::endl;
        
        // Method 1: Use bits64[0] (least significant 64 bits)
        uint64_t hash1 = point.x.bits64[0];
        std::cout << "Hash method 1 (bits64[0]): 0x" << std::hex << hash1 << std::endl;
        
        // Method 2: Use bits64[3] (most significant 64 bits)
        uint64_t hash2 = point.x.bits64[3];
        std::cout << "Hash method 2 (bits64[3]): 0x" << std::hex << hash2 << std::endl;
        
        // Method 3: XOR all parts
        uint64_t hash3 = point.x.bits64[0] ^ point.x.bits64[1] ^ point.x.bits64[2] ^ point.x.bits64[3];
        std::cout << "Hash method 3 (XOR all): 0x" << std::hex << hash3 << std::endl;
        
        // Test DP detection with different methods
        std::cout << "DP detection (mask 0x7):" << std::endl;
        std::cout << "  Method 1: " << (((hash1 & 0x7) == 0) ? "DP" : "NOT DP") << std::endl;
        std::cout << "  Method 2: " << (((hash2 & 0x7) == 0) ? "DP" : "NOT DP") << std::endl;
        std::cout << "  Method 3: " << (((hash3 & 0x7) == 0) ? "DP" : "NOT DP") << std::endl;
    }
    
    void debugSequentialKeys() {
        std::cout << "\n=== Sequential Keys Debug ===" << std::endl;
        
        std::cout << "Testing sequential private keys 1-10:" << std::endl;
        
        for (int i = 1; i <= 10; i++) {
            Int key;
            key.SetInt32(i);
            Point point = secp.ComputePublicKey(&key);
            
            uint64_t hash = point.x.bits64[0];
            bool is_dp = ((hash & 0x7) == 0);
            
            std::cout << "Key " << i << ": X[0]=0x" << std::hex << hash 
                      << ", DP=" << (is_dp ? "YES" : "NO") << std::dec << std::endl;
        }
    }
    
    void runAllDebugTests() {
        std::cout << "?? Point Structure and Hash Calculation Debug" << std::endl;
        std::cout << "=============================================" << std::endl;
        
        debugPointStructure();
        debugRandomGeneration();
        debugHashCalculation();
        debugSequentialKeys();
        
        std::cout << "\n?? DEBUG ANALYSIS COMPLETE" << std::endl;
        std::cout << "==========================" << std::endl;
        std::cout << "Review the output above to identify why DP detection is failing." << std::endl;
    }
};

int main() {
    std::cout << "Starting debug program..." << std::endl;
    try {
        PointStructureDebug debug;
        debug.runAllDebugTests();
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "Unknown exception occurred" << std::endl;
        return 1;
    }
    std::cout << "Debug program completed." << std::endl;
    return 0;
}
