#include "gecc.h"
#include "gecc/arith/ec.h"
#include "gecc/arith/fp.h"
#include "../SECPK1/Int.h"
#include "../Constants.h"
#include "GPUMath.h" // For jD, jPx, jPy constant arrays

// Define gECC types for secp256k1
DEFINE_FP(Secp256k1Fp, SECP256K1_FP, u32, 32, ColumnMajorLayout<1>, 8);
DEFINE_EC(Secp256k1, Jacobian, Secp256k1Fp, SECP256K1_EC, 2);

using GeccField = Secp256k1Fp;
using GeccEC = Secp256k1_Jacobian;
using GeccAffine = GeccEC::Affine;

// --- Helper device functions for data conversion ---

// Converts a point from the project's 64-bit limb format to gECC's 32-bit limb affine format
__device__ __forceinline__ GeccAffine toGeccAffine(const uint64_t* px, const uint64_t* py) {
    GeccAffine result;
    for (int i = 0; i < NB64BLOCK; i++) {
        result.x.digits[2 * i] = (uint32_t)px[i];
        result.x.digits[2 * i + 1] = (uint32_t)(px[i] >> 32);
        result.y.digits[2 * i] = (uint32_t)py[i];
        result.y.digits[2 * i + 1] = (uint32_t)(py[i] >> 32);
    }
    result.x = result.x.inplace_to_montgomery();
    result.y = result.y.inplace_to_montgomery();
    return result;
}

// Converts a point from gECC's affine format back to the project's format
__device__ __forceinline__ void fromGeccAffine(const GeccAffine& p, uint64_t* px, uint64_t* py) {
    GeccField x_normal = p.x.from_montgomery();
    GeccField y_normal = p.y.from_montgomery();
    for (int i = 0; i < NB64BLOCK; i++) {
        px[i] = (uint64_t)x_normal.digits[2 * i] | ((uint64_t)x_normal.digits[2 * i + 1] << 32);
        py[i] = (uint64_t)y_normal.digits[2 * i] | ((uint64_t)y_normal.digits[2 * i + 1] << 32);
    }
}

// --- Main GPU Kernel ---

extern "C" __global__ void gecc_kangaroo_kernel(
    uint64_t *kangaroos,
    uint32_t *found,
    uint32_t maxFound,
    uint64_t dpMask
) {
    // Calculate global thread and kangaroo indices
    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t localThreadId = threadIdx.x;

    // Each thread handles GPU_GRP_SIZE kangaroos
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint64_t kangaroo_idx = threadId * GPU_GRP_SIZE + g;

        // Pointer to the current kangaroo's data in global memory
        // KSIZE is the number of 64-bit words per kangaroo (4 for x, 4 for y, 2 for d, 1 for jump)
        uint64_t* kangaroo_ptr = kangaroos + kangaroo_idx * KSIZE;

        // Load kangaroo data into registers
        uint64_t px_orig[4], py_orig[4], dist_orig[2];
        for(int i=0; i<4; i++) px_orig[i] = kangaroo_ptr[i];
        for(int i=0; i<4; i++) py_orig[i] = kangaroo_ptr[4+i];
        for(int i=0; i<2; i++) dist_orig[i] = kangaroo_ptr[8+i];

        // Convert to gECC format
        GeccAffine p_current = toGeccAffine(px_orig, py_orig);

        // Perform NB_RUN random walk steps
        for (int run = 0; run < NB_RUN; run++) {
            // Determine jump
            uint32_t jmp_idx = (uint32_t)px_orig[0] & (NB_JUMP - 1);

            // Load jump point from constant memory
            uint64_t jpx_orig[4], jpy_orig[4];
            for(int i=0; i<4; i++) jpx_orig[i] = jPx[jmp_idx][i];
            for(int i=0; i<4; i++) jpy_orig[i] = jPy[jmp_idx][i];

            // Convert jump point to gECC format
            GeccAffine p_jump = toGeccAffine(jpx_orig, jpy_orig);

            // Perform the addition using gECC (the core of the optimization)
            p_current = p_current + p_jump;

            // Update distance (this part remains the same)
            Add128(dist_orig, jD[jmp_idx]);

            // Convert current point back to original format to get px_orig[0] for next jump
            fromGeccAffine(p_current, px_orig, py_orig);
        }

        // After all runs, store the final state back to global memory
        for(int i=0; i<4; i++) kangaroo_ptr[i] = px_orig[i];
        for(int i=0; i<4; i++) kangaroo_ptr[4+i] = py_orig[i];
        for(int i=0; i<2; i++) kangaroo_ptr[8+i] = dist_orig[i];

        // Check for distinguished point
        if (dpMask != 0 && (px_orig[3] & dpMask) == 0) {
            uint32_t pos = atomicAdd(found, 1);
            if (pos < maxFound) {
                // Write the DP to the output buffer
                // The format is: x[4], y[4], d[2], kIdx[2] (total 12 u64, or 24 u32)
                // ITEM_SIZE is 56 bytes (14 u32), but the output logic in GPUEngine seems to expect something different.
                // Replicating the logic from the original OutputDP for now.
                uint32_t* out_ptr = found + 1 + pos * 14; // 14 is ITEM_SIZE32
                uint64_t* out_ptr64 = (uint64_t*)out_ptr;

                for(int i=0; i<4; i++) out_ptr64[i] = px_orig[i];
                for(int i=0; i<2; i++) out_ptr64[4+i] = dist_orig[i]; // d is only 2 words
                out_ptr64[6] = kangaroo_idx;
            }
        }
    }
}
