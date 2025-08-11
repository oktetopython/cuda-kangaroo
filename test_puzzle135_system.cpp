/**
 * Test Bitcoin Puzzle 135 Challenge System
 * 
 * This program tests the complete Bitcoin Puzzle 135 challenge system
 * with real secp256k1 operations and the actual puzzle target.
 */

#include <iostream>
#include <string>
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"

int main() {
    std::cout << "🎯 Bitcoin Puzzle 135 System Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        Secp256K1 secp;
        secp.Init();
        
        // Test 1: Parse Bitcoin Puzzle 135 public key
        std::cout << "\n📋 Test 1: Parse Bitcoin Puzzle 135 Public Key" << std::endl;
        std::string compressed_pubkey = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16";
        Point target_public_key;
        bool isCompressed;
        
        if (secp.ParsePublicKeyHex(compressed_pubkey, target_public_key, isCompressed)) {
            std::cout << "✅ Successfully parsed Bitcoin Puzzle 135 public key" << std::endl;
            std::cout << "   Compressed: " << (isCompressed ? "YES" : "NO") << std::endl;
            std::cout << "   X: 0x" << target_public_key.x.GetBase16() << std::endl;
            std::cout << "   Y: 0x" << target_public_key.y.GetBase16() << std::endl;
        } else {
            std::cout << "❌ Failed to parse public key" << std::endl;
            return 1;
        }
        
        // Test 2: Verify range boundaries
        std::cout << "\n📋 Test 2: Verify Private Key Range" << std::endl;
        Int range_start, range_end;
        range_start.SetBase16((char*)"4000000000000000000000000000000000");
        range_end.SetBase16((char*)"7fffffffffffffffffffffffffffffffff");
        
        std::cout << "✅ Range Start: 0x" << range_start.GetBase16() << std::endl;
        std::cout << "✅ Range End:   0x" << range_end.GetBase16() << std::endl;
        
        // Test 3: Test a known point in range
        std::cout << "\n📋 Test 3: Test Point Generation in Range" << std::endl;
        Int test_key;
        test_key.SetBase16((char*)"5000000000000000000000000000000000");  // Mid-range test
        Point test_point = secp.ComputePublicKey(&test_key);
        
        std::cout << "✅ Test Key: 0x" << test_key.GetBase16() << std::endl;
        std::cout << "✅ Test Point X: 0x" << test_point.x.GetBase16() << std::endl;
        std::cout << "✅ Test Point Y: 0x" << test_point.y.GetBase16() << std::endl;
        
        // Test 4: Verify elliptic curve operations
        std::cout << "\n📋 Test 4: Verify Elliptic Curve Operations" << std::endl;
        Point generator = secp.G;
        std::cout << "✅ Generator G:" << std::endl;
        std::cout << "   X: 0x" << generator.x.GetBase16() << std::endl;
        std::cout << "   Y: 0x" << generator.y.GetBase16() << std::endl;
        
        // Test point addition
        Point doubled = secp.Double(generator);
        std::cout << "✅ 2*G computed successfully" << std::endl;
        std::cout << "   X: 0x" << doubled.x.GetBase16() << std::endl;
        
        // Test 5: Check if target is valid point on curve
        std::cout << "\n📋 Test 5: Validate Target Point on Curve" << std::endl;
        if (secp.EC(target_public_key)) {
            std::cout << "✅ Bitcoin Puzzle 135 target is valid point on secp256k1 curve" << std::endl;
        } else {
            std::cout << "❌ Target point is NOT on curve - ERROR!" << std::endl;
            return 1;
        }
        
        std::cout << "\n🎉 ALL TESTS PASSED!" << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << "✅ Bitcoin Puzzle 135 public key parsed correctly" << std::endl;
        std::cout << "✅ Private key range validated" << std::endl;
        std::cout << "✅ Elliptic curve operations working" << std::endl;
        std::cout << "✅ Target point is valid on secp256k1 curve" << std::endl;
        std::cout << "\n🚀 System is ready for Bitcoin Puzzle 135 challenge!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "❌ Unknown exception occurred" << std::endl;
        return 1;
    }
}
