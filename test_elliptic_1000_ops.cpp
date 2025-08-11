/**
 * Elliptic Curve 1000 Operations Test
 * 
 * This test specifically tests 1000 operations to match the comprehensive test
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
    cout << "🔍 Elliptic Curve 1000 Operations Test" << endl;
    cout << "=======================================" << endl;
    
    const int numOps = 1000; // Same as comprehensive test
    
    // Test 1: Legacy Implementation
    cout << "\n1. Testing Legacy Implementation (1000 ops):" << endl;
    Secp256K1 legacy;
    legacy.Init();
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < numOps; i++) {
        Int k;
        k.SetInt32(i + 1);
        Point P = legacy.ComputePublicKey(&k);
    }
    auto mid = high_resolution_clock::now();
    auto legacyTime = duration_cast<microseconds>(mid - start).count();
    
    cout << "Legacy: " << legacyTime << " μs for " << numOps << " operations" << endl;
    cout << "Average per operation: " << (double)legacyTime / numOps << " μs" << endl;
    
    // Test 2: Enhanced Implementation
    cout << "\n2. Testing Enhanced Implementation (1000 ops):" << endl;
    Secp256K1Enhanced enhanced;
    enhanced.Init();
    
    start = high_resolution_clock::now();
    for (int i = 0; i < numOps; i++) {
        Int k;
        k.SetInt32(i + 1);
        Point P = enhanced.ComputePublicKey(&k);
    }
    auto end = high_resolution_clock::now();
    auto enhancedTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Enhanced: " << enhancedTime << " μs for " << numOps << " operations" << endl;
    cout << "Average per operation: " << (double)enhancedTime / numOps << " μs" << endl;
    
    // Performance Analysis
    cout << "\n📊 Performance Analysis:" << endl;
    double ratio = (double)legacyTime / enhancedTime;
    cout << "Legacy vs Enhanced ratio: " << fixed << setprecision(2) << ratio << "x" << endl;
    
    if (ratio >= 0.9 && ratio <= 1.1) {
        cout << "✅ Enhanced implementation performs similarly to legacy" << endl;
    } else if (ratio > 1.1) {
        cout << "✅ Enhanced implementation is faster than legacy!" << endl;
    } else {
        cout << "⚠️  Enhanced implementation is slower than legacy" << endl;
    }
    
    // Test 3: Verify results are the same
    cout << "\n3. Result Verification:" << endl;
    Int testKey;
    testKey.SetInt32(12345);
    
    Point legacyResult = legacy.ComputePublicKey(&testKey);
    Point enhancedResult = enhanced.ComputePublicKey(&testKey);
    
    bool resultsMatch = (legacyResult.x.IsEqual(&enhancedResult.x) && 
                        legacyResult.y.IsEqual(&enhancedResult.y));
    cout << "Results match: " << (resultsMatch ? "YES" : "NO") << endl;
    
    if (resultsMatch) {
        cout << "✅ Enhanced implementation produces correct results" << endl;
    } else {
        cout << "❌ Enhanced implementation produces incorrect results!" << endl;
    }
    
    // Test 4: Performance breakdown
    cout << "\n4. Performance Breakdown:" << endl;
    cout << "Legacy per-op: " << fixed << setprecision(3) << (double)legacyTime / numOps << " μs" << endl;
    cout << "Enhanced per-op: " << fixed << setprecision(3) << (double)enhancedTime / numOps << " μs" << endl;
    
    if (enhancedTime > legacyTime * 2) {
        cout << "\n🐛 Significant Performance Issue Detected:" << endl;
        cout << "Enhanced implementation is more than 2x slower than legacy." << endl;
        cout << "This suggests a serious performance problem." << endl;
    } else if (enhancedTime > legacyTime * 1.5) {
        cout << "\n⚠️  Moderate Performance Issue:" << endl;
        cout << "Enhanced implementation has noticeable overhead." << endl;
    } else {
        cout << "\n✅ Performance is acceptable." << endl;
    }
    
    return 0;
}
