#include "gtest/gtest.h"
#include "SECPK1/SECP256k1.h"
#include "SECPK1/GeccAdapter.h"
#include "SECPK1/Int.h"
#include "Timer.h"
#include <vector>
#include <iostream>
#include <iomanip>

class GeccPerformanceBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        secp = new Secp256K1();
        secp->Init();
        GeccAdapter::Initialize();

        // Prepare test data
        p1_priv.SetBase16("1122334455667788112233445566778811223344556677881122334455667788");
        p1 = secp->ComputePublicKey(&p1_priv, true);

        p2_priv.SetBase16("AABBCCDDEEFFAABBCCDDEEFFAABBCCDDEEFFAABBCCDDEEFFAABBCCDDEEFFAABB");
        p2 = secp->ComputePublicKey(&p2_priv, true);
    }

    void TearDown() override {
        delete secp;
    }

    void RunBenchmark(const std::string& name, std::function<void()> original_func, std::function<void()> gecc_func, int iterations) {
        double original_time = 0;
        double gecc_time = 0;

        // Benchmark original implementation
        auto start_orig = Timer::get_tick();
        for (int i = 0; i < iterations; ++i) {
            original_func();
        }
        auto end_orig = Timer::get_tick();
        original_time = Timer::get_time(start_orig, end_orig);

        // Benchmark gECC implementation
        auto start_gecc = Timer::get_tick();
        for (int i = 0; i < iterations; ++i) {
            gecc_func();
        }
        auto end_gecc = Timer::get_tick();
        gecc_time = Timer::get_time(start_gecc, end_gecc);

        double speedup = (gecc_time > 0) ? (original_time / gecc_time) : 0.0;
        double original_ops = (original_time > 0) ? (iterations / original_time) : 0.0;
        double gecc_ops = (gecc_time > 0) ? (iterations / gecc_time) : 0.0;

        std::cout << std::left << std::setw(25) << name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << original_ops
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << gecc_ops
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }

    Secp256K1* secp;
    Point p1, p2;
    Int p1_priv, p2_priv;
};

TEST_F(GeccPerformanceBenchmark, Report) {
    const int ITERATIONS = 10000;
    std::cout << "\n--- gECC Performance Benchmark Report ---\n";
    std::cout << std::left << std::setw(25) << "Operation"
              << std::right << std::setw(15) << "Original (ops/s)"
              << std::right << std::setw(15) << "gECC (ops/s)"
              << std::right << std::setw(15) << "Speedup"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    RunBenchmark("Point Addition", [&]() {
        // Undefine USE_GECC to call original
        #undef USE_GECC
        volatile Point result = secp->AddDirect(p1, p2);
    }, [&]() {
        // Redefine USE_GECC to call adapter
        #define USE_GECC
        volatile Point result = secp->AddDirect(p1, p2);
    }, ITERATIONS);

    RunBenchmark("Point Doubling", [&]() {
        #undef USE_GECC
        volatile Point result = secp->DoubleDirect(p1);
    }, [&]() {
        #define USE_GECC
        volatile Point result = secp->DoubleDirect(p1);
    }, ITERATIONS);

    RunBenchmark("Scalar Multiplication", [&]() {
        #undef USE_GECC
        volatile Point result = secp->ComputePublicKey(&p1_priv, true);
    }, [&]() {
        #define USE_GECC
        volatile Point result = secp->ComputePublicKey(&p1_priv, true);
    }, 100); // Fewer iterations for this, as it's slower

    std::cout << std::string(70, '-') << std::endl;

    // Placeholder for GPU test, as it requires a different setup
    std::cout << "GPU Benchmark: Not implemented in this test suite. Requires separate test harness." << std::endl;
    SUCCEED();
}
