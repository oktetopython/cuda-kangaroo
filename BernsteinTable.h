/**
 * @file BernsteinTable.h
 * @brief Bernstein-Lange precomputed table for small discrete logarithm computation
 * 
 * Implementation based on "Computing Small Discrete Logarithms Faster" 
 * by Daniel J. Bernstein and Tanja Lange
 * 
 * Key features:
 * - Reduces DLP complexity from Θ(ℓ^1/2) to Θ(ℓ^1/3)
 * - Precomputed table size T = ℓ^1/3
 * - Optimized for small interval DLP problems
 */

#ifndef BERNSTEINTABLE_H
#define BERNSTEINTABLE_H

#include "SECPK1/SECP256k1.h"
#include "SECPK1/Int.h"
#include "SECPK1/Point.h"
#include <vector>
#include <unordered_map>
#include <string>

// Bernstein table configuration constants
#define BERNSTEIN_DEFAULT_TABLE_SIZE    (1 << 15)  // T = 2^15 = 32768
#define BERNSTEIN_DEFAULT_WALK_LENGTH   (1 << 16)  // W = 2^16 = 65536
#define BERNSTEIN_MAX_WALK_STEPS        (1 << 20)  // Maximum walk steps
#define BERNSTEIN_DP_BITS               16         // Distinguished point bits

/**
 * @brief Precomputed table entry for Bernstein-Lange algorithm
 */
struct BernsteinEntry {
    Point distinguished_point;    // Distinguished point (DP)
    Int   discrete_log;          // Corresponding discrete logarithm value
    uint32_t walk_length;        // Length of walk to reach this DP
    uint64_t hash_value;         // Hash of the distinguished point
    
    BernsteinEntry() : walk_length(0), hash_value(0) {}
    BernsteinEntry(const Point& dp, const Int& log, uint32_t length, uint64_t hash)
        : distinguished_point(dp), discrete_log(log), walk_length(length), hash_value(hash) {}
};

/**
 * @brief Bernstein-Lange precomputed table implementation
 * 
 * This class implements the precomputed table method from Bernstein-Lange paper
 * for accelerating small discrete logarithm computations.
 */
class BernsteinTable {
public:
    /**
     * @brief Constructor
     * @param secp Pointer to SECP256K1 curve instance
     * @param table_size Size of precomputed table (T parameter)
     * @param walk_length Expected walk length (W parameter)
     */
    BernsteinTable(Secp256K1* secp, uint32_t table_size = BERNSTEIN_DEFAULT_TABLE_SIZE, 
                   uint32_t walk_length = BERNSTEIN_DEFAULT_WALK_LENGTH);
    
    /**
     * @brief Destructor
     */
    ~BernsteinTable();
    
    /**
     * @brief Generate precomputed table for given interval
     * @param interval_start Start of the interval
     * @param interval_length Length of the interval (ℓ parameter)
     * @param base_point Base point g for DLP computation
     * @return true if table generation successful
     */
    bool GenerateTable(const Int& interval_start, const Int& interval_length, const Point& base_point);
    
    /**
     * @brief Solve discrete logarithm using precomputed table
     * @param target_point Target point h for which to find log_g(h)
     * @param base_point Base point g
     * @param interval_start Start of search interval
     * @param interval_length Length of search interval
     * @param result Output parameter for the discrete logarithm
     * @return true if DLP solved successfully
     */
    bool SolveDLP(const Point& target_point, const Point& base_point, 
                  const Int& interval_start, const Int& interval_length, Int& result);
    
    /**
     * @brief Check if a point is a distinguished point
     * @param point Point to check
     * @return true if point is distinguished
     */
    bool IsDistinguishedPoint(const Point& point) const;
    
    /**
     * @brief Compute hash value for a point
     * @param point Input point
     * @return 64-bit hash value
     */
    uint64_t ComputePointHash(const Point& point) const;
    
    /**
     * @brief Perform r-adding walk from starting point
     * @param start_point Starting point for the walk
     * @param start_log Starting discrete logarithm value
     * @param max_steps Maximum number of steps
     * @param final_point Output final point reached
     * @param final_log Output final discrete logarithm
     * @return Number of steps taken
     */
    uint32_t PerformWalk(const Point& start_point, const Int& start_log, uint32_t max_steps,
                         Point& final_point, Int& final_log);
    
    /**
     * @brief Save precomputed table to file
     * @param filename Output filename
     * @return true if save successful
     */
    bool SaveTable(const std::string& filename) const;
    
    /**
     * @brief Load precomputed table from file
     * @param filename Input filename
     * @return true if load successful
     */
    bool LoadTable(const std::string& filename);
    
    /**
     * @brief Get table statistics
     * @return String containing table information
     */
    std::string GetTableInfo() const;
    
    /**
     * @brief Clear the precomputed table
     */
    void ClearTable();
    
    /**
     * @brief Get table size
     * @return Number of entries in the table
     */
    uint32_t GetTableSize() const { return table_size; }
    
    /**
     * @brief Get number of entries currently in table
     * @return Number of valid entries
     */
    uint32_t GetEntryCount() const { return entry_count; }
    
    /**
     * @brief Verify table integrity
     * @return true if table is valid
     */
    bool VerifyTable() const;

private:
    Secp256K1* secp;                                    // SECP256K1 curve instance
    uint32_t table_size;                                // Size of precomputed table (T)
    uint32_t walk_length;                               // Expected walk length (W)
    uint32_t entry_count;                               // Current number of entries
    uint32_t dp_bits;                                   // Distinguished point bits
    
    std::vector<BernsteinEntry> table;                  // Precomputed table entries
    std::unordered_map<uint64_t, uint32_t> hash_index; // Hash to index mapping
    
    // Jump table for r-adding walks (precomputed jump distances)
    std::vector<Point> jump_points;                     // Precomputed jump points
    std::vector<Int> jump_distances;                    // Corresponding jump distances
    uint32_t jump_table_size;                           // Size of jump table
    
    /**
     * @brief Initialize jump table for r-adding walks
     * @param base_point Base point for jump computation
     */
    void InitializeJumpTable(const Point& base_point);
    
    /**
     * @brief Select next jump in r-adding walk
     * @param current_point Current point in walk
     * @param jump_point Output jump point
     * @param jump_distance Output jump distance
     */
    void SelectJump(const Point& current_point, Point& jump_point, Int& jump_distance) const;
    
    /**
     * @brief Generate candidate distinguished points
     * @param interval_start Start of interval
     * @param interval_length Length of interval
     * @param base_point Base point
     * @param candidates Output vector of candidate entries
     */
    void GenerateCandidates(const Int& interval_start, const Int& interval_length, 
                           const Point& base_point, std::vector<BernsteinEntry>& candidates);
    
    /**
     * @brief Select best entries for final table based on weights
     * @param candidates Input candidate entries
     */
    void SelectBestEntries(std::vector<BernsteinEntry>& candidates);
    
    /**
     * @brief Calculate weight for a distinguished point
     * @param entry Table entry to evaluate
     * @param visit_count Number of times this DP was visited
     * @param total_walk_length Total walk length to reach this DP
     * @return Weight value (higher is better)
     */
    double CalculateWeight(const BernsteinEntry& entry, uint32_t visit_count, 
                          uint32_t total_walk_length) const;
};

#endif // BERNSTEINTABLE_H
