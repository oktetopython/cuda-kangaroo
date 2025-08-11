/*
* Unified Elliptic Curve Operations for SECP256K1
* Eliminates duplicate EC implementations across the project
*/

#ifndef UNIFIED_ELLIPTIC_CURVE_H
#define UNIFIED_ELLIPTIC_CURVE_H

#include "UnifiedGPUMath.h"

// SECP256K1 curve parameters
__device__ __constant__ uint64_t CURVE_B[5] = {7, 0, 0, 0, 0}; // y² = x³ + 7
__device__ __constant__ uint64_t CURVE_ORDER[5] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL, 0ULL
};

// Point representation modes
enum PointMode {
    AFFINE,      // (x, y)
    JACOBIAN,    // (X, Y, Z) where x = X/Z², y = Y/Z³
    MONTGOMERY   // (X, Z) for Montgomery ladder
};

class UnifiedEllipticCurve {
public:
    // Unified point addition in affine coordinates
    __device__ static void PointAddAffine(
        uint64_t *rx, uint64_t *ry,           // Result point
        uint64_t *px, uint64_t *py,           // Point P
        uint64_t *qx, uint64_t *qy            // Point Q
    ) {
        uint64_t dx[5], dy[5], lambda[5], lambda_sq[5];
        uint64_t temp1[5], temp2[5];
        
        // Check for point at infinity cases
        if(_IsZero(px) && _IsZero(py)) {
            UnifiedGPUMath::Load256(rx, qx);
            UnifiedGPUMath::Load256(ry, qy);
            return;
        }
        if(_IsZero(qx) && _IsZero(qy)) {
            UnifiedGPUMath::Load256(rx, px);
            UnifiedGPUMath::Load256(ry, py);
            return;
        }
        
        // Check if points are equal (point doubling case)
        if(_IsEqual(px, qx)) {
            if(_IsEqual(py, qy)) {
                PointDoubleAffine(rx, ry, px, py);
                return;
            } else {
                // Points are inverses, result is point at infinity
                UnifiedGPUMath::SetZero(rx);
                UnifiedGPUMath::SetZero(ry);
                return;
            }
        }
        
        // Standard point addition
        // λ = (qy - py) / (qx - px)
        UnifiedGPUMath::ModSub256(dy, qy, py);
        UnifiedGPUMath::ModSub256(dx, qx, px);
        UnifiedGPUMath::ModInv256(dx, dx);
        UnifiedGPUMath::ModMult256(lambda, dy, dx);
        
        // rx = λ² - px - qx
        UnifiedGPUMath::ModSqr256(lambda_sq, lambda);
        UnifiedGPUMath::ModSub256(temp1, lambda_sq, px);
        UnifiedGPUMath::ModSub256(rx, temp1, qx);
        
        // ry = λ(px - rx) - py
        UnifiedGPUMath::ModSub256(temp1, px, rx);
        UnifiedGPUMath::ModMult256(temp2, lambda, temp1);
        UnifiedGPUMath::ModSub256(ry, temp2, py);
    }
    
    // Unified point doubling in affine coordinates
    __device__ static void PointDoubleAffine(
        uint64_t *rx, uint64_t *ry,           // Result point
        uint64_t *px, uint64_t *py            // Point P
    ) {
        uint64_t lambda[5], lambda_sq[5];
        uint64_t temp1[5], temp2[5], temp3[5];
        uint64_t three[5] = {3, 0, 0, 0, 0};
        uint64_t two[5] = {2, 0, 0, 0, 0};
        
        // Check for point at infinity
        if(_IsZero(px) && _IsZero(py)) {
            UnifiedGPUMath::SetZero(rx);
            UnifiedGPUMath::SetZero(ry);
            return;
        }
        
        // λ = (3px² + a) / (2py), where a = 0 for secp256k1
        UnifiedGPUMath::ModSqr256(temp1, px);           // px²
        UnifiedGPUMath::ModMult256(temp2, three, temp1); // 3px²
        
        UnifiedGPUMath::ModMult256(temp3, two, py);      // 2py
        UnifiedGPUMath::ModInv256(temp3, temp3);         // 1/(2py)
        UnifiedGPUMath::ModMult256(lambda, temp2, temp3); // λ
        
        // rx = λ² - 2px
        UnifiedGPUMath::ModSqr256(lambda_sq, lambda);
        UnifiedGPUMath::ModMult256(temp1, two, px);      // 2px
        UnifiedGPUMath::ModSub256(rx, lambda_sq, temp1);
        
        // ry = λ(px - rx) - py
        UnifiedGPUMath::ModSub256(temp1, px, rx);
        UnifiedGPUMath::ModMult256(temp2, lambda, temp1);
        UnifiedGPUMath::ModSub256(ry, temp2, py);
    }
    
    // Optimized point addition with batch inversion
    __device__ static void BatchPointAdd(
        uint64_t px[][4], uint64_t py[][4],   // Input points P
        uint64_t qx[][4], uint64_t qy[][4],   // Input points Q  
        uint64_t rx[][4], uint64_t ry[][4],   // Result points
        int count                             // Number of points
    ) {
        uint64_t dx[32][4]; // Denominators for batch inversion
        uint64_t dy[32][4]; // Numerators
        
        // Phase 1: Compute all denominators
        for(int i = 0; i < count; i++) {
            UnifiedGPUMath::ModSub256(dx[i], qx[i], px[i]);
            UnifiedGPUMath::ModSub256(dy[i], qy[i], py[i]);
        }
        
        // Phase 2: Batch inversion of denominators
        UnifiedGPUMath::BatchModInv(dx, count);
        
        // Phase 3: Complete point additions
        for(int i = 0; i < count; i++) {
            uint64_t lambda[5], lambda_sq[5], temp1[5], temp2[5];
            
            // λ = dy[i] * dx[i] (dx[i] is now inverted)
            UnifiedGPUMath::ModMult256(lambda, dy[i], dx[i]);
            
            // rx = λ² - px - qx
            UnifiedGPUMath::ModSqr256(lambda_sq, lambda);
            UnifiedGPUMath::ModSub256(temp1, lambda_sq, px[i]);
            UnifiedGPUMath::ModSub256(rx[i], temp1, qx[i]);
            
            // ry = λ(px - rx) - py
            UnifiedGPUMath::ModSub256(temp1, px[i], rx[i]);
            UnifiedGPUMath::ModMult256(temp2, lambda, temp1);
            UnifiedGPUMath::ModSub256(ry[i], temp2, py[i]);
        }
    }
    
    // Scalar multiplication using Montgomery ladder
    __device__ static void ScalarMult(
        uint64_t *rx, uint64_t *ry,           // Result point
        uint64_t *scalar,                     // Scalar (256-bit)
        uint64_t *px, uint64_t *py            // Base point
    ) {
        uint64_t r0x[5], r0y[5]; // Point R0
        uint64_t r1x[5], r1y[5]; // Point R1
        
        // Initialize: R0 = O (point at infinity), R1 = P
        UnifiedGPUMath::SetZero(r0x);
        UnifiedGPUMath::SetZero(r0y);
        UnifiedGPUMath::Load256(r1x, px);
        UnifiedGPUMath::Load256(r1y, py);
        
        // Montgomery ladder
        for(int i = 255; i >= 0; i--) {
            int bit = GetBit(scalar, i);
            
            if(bit == 0) {
                // R1 = R0 + R1, R0 = 2*R0
                PointAddAffine(r1x, r1y, r0x, r0y, r1x, r1y);
                PointDoubleAffine(r0x, r0y, r0x, r0y);
            } else {
                // R0 = R0 + R1, R1 = 2*R1
                PointAddAffine(r0x, r0y, r0x, r0y, r1x, r1y);
                PointDoubleAffine(r1x, r1y, r1x, r1y);
            }
        }
        
        UnifiedGPUMath::Load256(rx, r0x);
        UnifiedGPUMath::Load256(ry, r0y);
    }
    
    // Optimized Kangaroo jump operation
    __device__ static void KangarooJump(
        uint64_t *rx, uint64_t *ry,           // Result kangaroo position
        uint64_t *px, uint64_t *py,           // Current kangaroo position
        uint64_t *jx, uint64_t *jy,           // Jump point
        uint64_t *distance                    // Distance to add
    ) {
        // This is the core operation in Kangaroo algorithm
        // Equivalent to: (rx, ry) = (px, py) + (jx, jy)
        PointAddAffine(rx, ry, px, py, jx, jy);
    }
    
    // Batch Kangaroo jumps with optimized inversion
    __device__ static void BatchKangarooJump(
        uint64_t px[][4], uint64_t py[][4],   // Current positions
        uint64_t jx[][4], uint64_t jy[][4],   // Jump points
        uint64_t rx[][4], uint64_t ry[][4],   // Result positions
        int count                             // Number of kangaroos
    ) {
        BatchPointAdd(px, py, jx, jy, rx, ry, count);
    }
    
    // Point validation
    __device__ static bool IsValidPoint(uint64_t *px, uint64_t *py) {
        uint64_t left[5], right[5], temp[5];
        
        // Check if point is at infinity
        if(_IsZero(px) && _IsZero(py)) return true;
        
        // Check if y² = x³ + 7 (mod p)
        UnifiedGPUMath::ModSqr256(left, py);              // y²
        UnifiedGPUMath::ModSqr256(temp, px);              // x²
        UnifiedGPUMath::ModMult256(right, temp, px);      // x³
        UnifiedGPUMath::Add256(right, right, CURVE_B);    // x³ + 7
        
        return _IsEqual(left, right);
    }

private:
    // Helper function to get bit from 256-bit scalar
    __device__ static int GetBit(uint64_t *scalar, int bit_index) {
        int word_index = bit_index / 64;
        int bit_offset = bit_index % 64;
        return (scalar[word_index] >> bit_offset) & 1;
    }
};

// Legacy compatibility functions
#define PointAdd(rx,ry,px,py,qx,qy) UnifiedEllipticCurve::PointAddAffine(rx,ry,px,py,qx,qy)
#define PointDouble(rx,ry,px,py) UnifiedEllipticCurve::PointDoubleAffine(rx,ry,px,py)

#endif // UNIFIED_ELLIPTIC_CURVE_H