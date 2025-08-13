#include "gtest/gtest.h"
#include "SECPK1/SECP256k1.h"
#include "SECPK1/GeccAdapter.h"
#include "SECPK1/Int.h"

class GeccAdapterTest : public ::testing::Test {
protected:
    void SetUp() override {
        secp = new Secp256K1();
        secp->Init();
        GeccAdapter::Initialize();
    }

    void TearDown() override {
        delete secp;
    }

    Secp256K1* secp;
};

// Undefine USE_GECC to access the original implementations for comparison
#ifdef USE_GECC
#undef USE_GECC
#endif

TEST_F(GeccAdapterTest, Add) {
    Int p1_priv;
    p1_priv.SetBase16("1122334455667788112233445566778811223344556677881122334455667788");
    Point p1 = secp->ComputePublicKey(&p1_priv, true);

    Int p2_priv;
    p2_priv.SetBase16("AABBCCDDEEFFAABBCCDDEEFFAABBCCDDEEFFAABBCCDDEEFFAABBCCDDEEFFAABB");
    Point p2 = secp->ComputePublicKey(&p2_priv, true);

    // Re-define USE_GECC to test the adapter
    #define USE_GECC

    Point sum_gecc = GeccAdapter::Add(p1, p2);

    // Undefine again to call original implementation
    #undef USE_GECC

    Point sum_original = secp->AddDirect(p1, p2);

    ASSERT_TRUE(sum_gecc.equals(sum_original));
}

TEST_F(GeccAdapterTest, Double) {
    Int p1_priv;
    p1_priv.SetBase16("1122334455667788112233445566778811223344556677881122334455667788");
    Point p1 = secp->ComputePublicKey(&p1_priv, true);

    #define USE_GECC
    Point double_gecc = GeccAdapter::Double(p1);
    #undef USE_GECC

    Point double_original = secp->DoubleDirect(p1);

    ASSERT_TRUE(double_gecc.equals(double_original));
}

TEST_F(GeccAdapterTest, ScalarMult) {
    Int p1_priv;
    p1_priv.SetBase16("1122334455667788112233445566778811223344556677881122334455667788");

    #define USE_GECC
    Point p_gecc = secp->ComputePublicKey(&p1_priv, true);
    #undef USE_GECC

    Point p_original = secp->ComputePublicKey(&p1_priv, true);

    ASSERT_TRUE(p_gecc.equals(p_original));
}
