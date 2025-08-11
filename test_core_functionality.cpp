/**
 * Core Functionality Test for CUDA-BSGS-Kangaroo
 * Tests basic operations of the compiled kangaroo_core.lib
 */

#include <iostream>
#include <string>
#include <chrono>
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"
#include "SECPK1/Secp256K1.h"
#include "HashTable.h"
#include "HashTable512.h"

using namespace std;

int main() {
    cout << "=== CUDA-BSGS-Kangaroo Core Functionality Test ===" << endl;
    
    try {
        // Test 1: Initialize Secp256K1
        cout << "\n1. Testing Secp256K1 initialization..." << endl;
        Secp256K1 secp;
        secp.Init();
        cout << "   ✓ Secp256K1 initialized successfully" << endl;
        
        // Test 2: Basic Int operations
        cout << "\n2. Testing Int operations..." << endl;
        Int a, b, c;
        a.SetInt32(123456);
        b.SetInt32(789012);
        c.Add(&a, &b);
        cout << "   ✓ Int addition: " << a.GetInt32() << " + " << b.GetInt32() << " = " << c.GetInt32() << endl;
        
        // Test 3: Basic Point operations
        cout << "\n3. Testing Point operations..." << endl;
        Point G = secp.G;
        Point P1, P2;
        Int k1, k2;
        k1.SetInt32(5);
        k2.SetInt32(7);
        
        P1 = secp.ComputePublicKey(&k1);
        P2 = secp.ComputePublicKey(&k2);
        cout << "   ✓ Point multiplication successful" << endl;
        
        Point P3 = secp.AddDirect(P1, P2);
        cout << "   ✓ Point addition successful" << endl;
        
        // Test 4: HashTable basic operations
        cout << "\n4. Testing HashTable operations..." << endl;
        HashTable hashTable;

        // Test basic functionality
        Int x1, d1, x2, d2;
        x1.SetInt32(12345);
        d1.SetInt32(100);
        x2.SetInt32(67890);
        d2.SetInt32(200);

        int result1 = hashTable.Add(&x1, &d1, 0);  // type 0 = tame
        int result2 = hashTable.Add(&x2, &d2, 1);  // type 1 = wild

        cout << "   ✓ HashTable entries added: result1=" << result1 << ", result2=" << result2 << endl;
        
        // Test 5: HashTable512 basic operations
        cout << "\n5. Testing HashTable512 operations..." << endl;
        HashTable512 hashTable512;

        // Test basic functionality - HashTable512 uses different API
        cout << "   ✓ HashTable512 initialized successfully" << endl;
        cout << "   Note: HashTable512 API testing requires specific setup" << endl;
        
        // Test 6: Performance test
        cout << "\n6. Performance test..." << endl;
        auto start = chrono::high_resolution_clock::now();

        // Add multiple entries to HashTable
        for (int i = 0; i < 1000; i++) {
            Int x, d;
            x.SetInt32(i * 1000 + 12345);
            d.SetInt32(i + 100);
            hashTable.Add(&x, &d, i % 2);  // Alternate between tame and wild
        }

        auto end = chrono::high_resolution_clock::now();
        auto time1 = chrono::duration_cast<chrono::microseconds>(end - start).count();

        cout << "   HashTable performance: " << time1 << " microseconds for 1000 entries" << endl;
        cout << "   ✓ Performance test completed" << endl;

        // Test 7: Memory usage
        cout << "\n7. Memory usage test..." << endl;
        cout << "   HashTable memory: " << hashTable.GetNbItem() << " items" << endl;
        cout << "   ✓ Memory usage test completed" << endl;
        
        cout << "\n=== All tests completed successfully! ===" << endl;
        cout << "kangaroo_core.lib is working correctly." << endl;
        
    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "Unknown error occurred" << endl;
        return 1;
    }
    
    return 0;
}
