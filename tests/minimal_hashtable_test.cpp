/**
 * @file minimal_hashtable_test.cpp
 * @brief Minimal test to isolate HashTable crash
 */

#include <gtest/gtest.h>
#include <iostream>

// NEW: 最小化测试，逐步排除依赖
#include "../HashTable.h"

namespace kangaroo_test {

/**
 * @brief Minimal HashTable test
 */
TEST(MinimalHashTableTest, BasicConstruction) {
    std::cout << "Test starting..." << std::endl;
    
    try {
        std::cout << "Creating HashTable..." << std::endl;
        HashTable hash_table;
        std::cout << "HashTable created successfully" << std::endl;
        
        std::cout << "Getting item count..." << std::endl;
        uint64_t count = hash_table.GetNbItem();
        std::cout << "Item count: " << count << std::endl;
        
        EXPECT_EQ(count, 0);
        std::cout << "Test completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        FAIL() << "Exception in HashTable construction: " << e.what();
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        FAIL() << "Unknown exception in HashTable construction";
    }
}

/**
 * @brief Test with Int creation
 */
TEST(MinimalHashTableTest, WithIntCreation) {
    std::cout << "Test WithIntCreation starting..." << std::endl;
    
    try {
        HashTable hash_table;
        std::cout << "HashTable created" << std::endl;
        
        // Try to create Int objects
        Int x, d;
        std::cout << "Int objects created" << std::endl;
        
        // Try simple operations without random
        x.SetInt32(123);
        d.SetInt32(456);
        std::cout << "Int values set" << std::endl;
        
        int result = hash_table.Add(&x, &d, 0);
        std::cout << "Add result: " << result << std::endl;
        
        EXPECT_TRUE(result == ADD_OK || result == ADD_DUPLICATE || result == ADD_COLLISION);
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        FAIL() << "Exception in WithIntCreation: " << e.what();
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        FAIL() << "Unknown exception in WithIntCreation";
    }
}

} // namespace kangaroo_test
