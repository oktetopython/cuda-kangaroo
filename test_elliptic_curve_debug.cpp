/**
 * Elliptic Curve Performance Debug Test
 * 
 * This test diagnoses the performance issues in Secp256K1Enhanced
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"
#include "SECPK1/Secp256K1.h"
#include "gEccAdapter.h"

using namespace std;
using namespace std::chrono;

int main() {
    cout << "🔍 Elliptic Curve Performance Debug Test" << endl;
    cout << "=========================================" << endl;
    
    const int numOps = 100; // Smaller number for debugging
    
    // Test 1: Direct Legacy Implementation
    cout << "\n1. Testing Direct Legacy Implementation:" << endl;
    Secp256K1 legacy;
    legacy.Init();
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < numOps; i++) {
        Int k;
        k.SetInt32(i + 1);
        Point P = legacy.ComputePublicKey(&k);
    }
    auto end = high_resolution_clock::now();
    auto legacyTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Direct Legacy: " << legacyTime << " μs for " << numOps << " operations" << endl;
    cout << "Average per operation: " << (double)legacyTime / numOps << " μs" << endl;
    
    // Test 2: Enhanced Implementation (should call legacy)
    cout << "\n2. Testing Enhanced Implementation:" << endl;
    Secp256K1Enhanced enhanced;
    enhanced.Init();
    
    start = high_resolution_clock::now();
    for (int i = 0; i < numOps; i++) {
        Int k;
        k.SetInt32(i + 1);
        Point P = enhanced.ComputePublicKey(&k);
    }
    end = high_resolution_clock::now();
    auto enhancedTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Enhanced: " << enhancedTime << " μs for " << numOps << " operations" << endl;
    cout << "Average per operation: " << (double)enhancedTime / numOps << " μs" << endl;
    
    // Test 3: Single operation comparison
    cout << "\n3. Single Operation Comparison:" << endl;
    Int testKey;
    testKey.SetInt32(12345);
    
    start = high_resolution_clock::now();
    Point legacyResult = legacy.ComputePublicKey(&testKey);
    auto legacySingle = high_resolution_clock::now();
    
    Point enhancedResult = enhanced.ComputePublicKey(&testKey);
    auto enhancedSingle = high_resolution_clock::now();
    
    auto legacySingleTime = duration_cast<microseconds>(legacySingle - start).count();
    auto enhancedSingleTime = duration_cast<microseconds>(enhancedSingle - legacySingle).count();
    
    cout << "Legacy single operation: " << legacySingleTime << " μs" << endl;
    cout << "Enhanced single operation: " << enhancedSingleTime << " μs" << endl;
    
    // Test 4: Verify results are the same
    cout << "\n4. Result Verification:" << endl;
    cout << "Legacy result X: " << legacyResult.x.GetBase16() << endl;
    cout << "Enhanced result X: " << enhancedResult.x.GetBase16() << endl;
    
    bool resultsMatch = (legacyResult.x.IsEqual(&enhancedResult.x) && 
                        legacyResult.y.IsEqual(&enhancedResult.y));
    cout << "Results match: " << (resultsMatch ? "YES" : "NO") << endl;
    
    // Test 5: Batch operation test
    cout << "\n5. Batch Operation Test:" << endl;
    vector<Int> keys;
    for (int i = 0; i < numOps; i++) {
        Int k;
        k.SetInt32(i + 1);
        keys.push_back(k);
    }
    
    start = high_resolution_clock::now();
    vector<Point> batchResults = enhanced.ComputePublicKeys(keys);
    end = high_resolution_clock::now();
    auto batchTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Batch operation: " << batchTime << " μs for " << numOps << " operations" << endl;
    cout << "Average per operation: " << (double)batchTime / numOps << " μs" << endl;
    
    // Performance Analysis
    cout << "\n📊 Performance Analysis:" << endl;
    double overhead = (double)enhancedTime / legacyTime;
    cout << "Enhanced vs Legacy ratio: " << fixed << setprecision(2) << overhead << "x" << endl;
    
    if (overhead > 1.5) {
        cout << "⚠️  PERFORMANCE ISSUE DETECTED!" << endl;
        cout << "Enhanced implementation is significantly slower than legacy." << endl;
        cout << "This suggests overhead in the Enhanced wrapper." << endl;
    } else if (overhead < 0.8) {
        cout << "✅ PERFORMANCE IMPROVEMENT CONFIRMED!" << endl;
        cout << "Enhanced implementation is faster than legacy." << endl;
    } else {
        cout << "➡️  Performance is similar (within expected overhead range)." << endl;
    }
    
    // Detailed timing breakdown
    cout << "\n🔬 Detailed Timing Breakdown:" << endl;
    cout << "Legacy per-op: " << (double)legacyTime / numOps << " μs" << endl;
    cout << "Enhanced per-op: " << (double)enhancedTime / numOps << " μs" << endl;
    cout << "Batch per-op: " << (double)batchTime / numOps << " μs" << endl;
    
    if (enhancedTime > legacyTime) {
        cout << "\n🐛 Potential Issues:" << endl;
        cout << "- Enhanced wrapper may have unnecessary overhead" << endl;
        cout << "- Multiple initialization calls" << endl;
        cout << "- Inefficient fallback mechanism" << endl;
        cout << "- Memory allocation overhead" << endl;
    }
    
    return 0;
}
