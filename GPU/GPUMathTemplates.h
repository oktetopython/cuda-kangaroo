/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GPUMATHTEMPLATES_H
#define GPUMATHTEMPLATES_H

// OPTIMIZATION: Template-based GPU math operations to reduce code duplication
// This file provides templated versions of common mathematical operations
// to replace the hand-unrolled repetitive code in GPUMath.h

// Forward declare required macros (these should be defined in GPUMath.h)
#ifndef UADDO1
#define UADDO1(c, a) asm volatile("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define UADDC1(c, a) asm volatile("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define UADD1(c, a) asm volatile("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));
#define USUB(c, a, b) asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define USUBB(c, a, b) asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define UADD(c, a, b) asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define UADDC(c, a, b) asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define ULT(a, b) ((a) < (b))
#define UMULLO(r, a, b) asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
#define UMULHI(r, a, b) asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
#endif

// ---------------------------------------------------------------------------------------
// Template for unsigned multiplication with different operand types
// ---------------------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ void UMultTemplate(uint64_t *r, const uint64_t *a, T b)
{
  if constexpr (sizeof(T) == sizeof(uint64_t))
  {
    // 64-bit operand case
    UMULLO(r[0], a[0], b);
    UMULLO(r[1], a[1], b);
    UMULLO(r[2], a[2], b);
    UMULLO(r[3], a[3], b);
    UMULHI(r[4], a[3], b);
    UMULHI(r[3], a[2], b);
    UMULHI(r[2], a[1], b);
    UMULHI(r[1], a[0], b);
  }
  else
  {
    // Array operand case - delegate to specialized version
    UMultArrayTemplate(r, a, b);
  }
}

template <>
__device__ __forceinline__ void UMultTemplate<const uint64_t *>(uint64_t *r, const uint64_t *a, const uint64_t *b)
{
  // 256*256 bit multiplication template
  uint64_t t[5];

  // First multiplication: a * b[0]
  UMultTemplate(r, a, b[0]);

  // Second multiplication: a * b[1], shifted by 64 bits
  UMultTemplate(t, a, b[1]);
  UADDO1(r[1], t[0]);
  UADDC1(r[2], t[1]);
  UADDC1(r[3], t[2]);
  UADDC1(r[4], t[3]);
  UADD1(r[5], t[4]);

  // Third multiplication: a * b[2], shifted by 128 bits
  UMultTemplate(t, a, b[2]);
  UADDO1(r[2], t[0]);
  UADDC1(r[3], t[1]);
  UADDC1(r[4], t[2]);
  UADDC1(r[5], t[3]);
  UADD1(r[6], t[4]);

  // Fourth multiplication: a * b[3], shifted by 192 bits
  UMultTemplate(t, a, b[3]);
  UADDO1(r[3], t[0]);
  UADDC1(r[4], t[1]);
  UADDC1(r[5], t[2]);
  UADDC1(r[6], t[3]);
  UADD1(r[7], t[4]);
}

// ---------------------------------------------------------------------------------------
// Template for modular negation operations
// ---------------------------------------------------------------------------------------

template <bool InPlace = false>
__device__ __forceinline__ void ModNeg256Template(uint64_t *r, const uint64_t *a = nullptr)
{
  uint64_t t[4];

  if constexpr (InPlace)
  {
    // In-place negation: r = -r mod p
    t[0] = (~r[0]) + 1;
    t[1] = (~r[1]) + (t[0] == 0 ? 1 : 0);
    t[2] = (~r[2]) + ((t[1] == 0 && t[0] == 0) ? 1 : 0);
    t[3] = (~r[3]) + ((t[2] == 0 && t[1] == 0 && t[0] == 0) ? 1 : 0);
  }
  else
  {
    // Two-operand negation: r = -a mod p
    t[0] = (~a[0]) + 1;
    t[1] = (~a[1]) + (t[0] == 0 ? 1 : 0);
    t[2] = (~a[2]) + ((t[1] == 0 && t[0] == 0) ? 1 : 0);
    t[3] = (~a[3]) + ((t[2] == 0 && t[1] == 0 && t[0] == 0) ? 1 : 0);
  }

  // Subtract from field prime p = 2^256 - 2^32 - 977
  USUB(t[0], 0xFFFFFFFEFFFFFC2FULL);
  USUBB(t[1], 0xFFFFFFFFFFFFFFFFULL);
  USUBB(t[2], 0xFFFFFFFFFFFFFFFFULL);
  USUBB(t[3], 0xFFFFFFFFFFFFFFFFULL);

  r[0] = t[0];
  r[1] = t[1];
  r[2] = t[2];
  r[3] = t[3];
}

// ---------------------------------------------------------------------------------------
// Template for modular subtraction operations
// ---------------------------------------------------------------------------------------

template <bool InPlace = false>
__device__ __forceinline__ void ModSub256Template(uint64_t *r, const uint64_t *a, const uint64_t *b = nullptr)
{
  uint64_t t;

  if constexpr (InPlace)
  {
    // In-place subtraction: r = r - a mod p
    USUB(r[0], a[0]);
    USUBB(r[1], a[1]);
    USUBB(r[2], a[2]);
    USUBB(r[3], a[3]);

    if (ULT(r[0], a[0]))
    {
      // Borrow occurred, add field prime
      UADD(r[0], 0xFFFFFFFEFFFFFC2FULL);
      UADDC(r[1], 0xFFFFFFFFFFFFFFFFULL);
      UADDC(r[2], 0xFFFFFFFFFFFFFFFFULL);
      UADDC(r[3], 0xFFFFFFFFFFFFFFFFULL);
    }
  }
  else
  {
    // Two-operand subtraction: r = a - b mod p
    USUB(r[0], a[0], b[0]);
    USUBB(r[1], a[1], b[1]);
    USUBB(r[2], a[2], b[2]);
    USUBB(r[3], a[3], b[3]);

    if (ULT(a[0], b[0]))
    {
      // Borrow occurred, add field prime
      UADD(r[0], 0xFFFFFFFEFFFFFC2FULL);
      UADDC(r[1], 0xFFFFFFFFFFFFFFFFULL);
      UADDC(r[2], 0xFFFFFFFFFFFFFFFFULL);
      UADDC(r[3], 0xFFFFFFFFFFFFFFFFULL);
    }
  }
}

// ---------------------------------------------------------------------------------------
// Template for modular multiplication with reduction
// ---------------------------------------------------------------------------------------

template <bool InPlace = false>
__device__ __forceinline__ void ModMultTemplate(uint64_t *r, const uint64_t *a, const uint64_t *b = nullptr)
{
  uint64_t r512[8];
  uint64_t t[5];

  if constexpr (InPlace)
  {
    // In-place multiplication: r = r * a mod p
    // Store original r values for multiplication
    uint64_t orig_r[4] = {r[0], r[1], r[2], r[3]};

    // 256*256 bit multiplication
    UMultTemplate(r512, orig_r, a);
  }
  else
  {
    // Two-operand multiplication: r = a * b mod p
    UMultTemplate(r512, a, b);
  }

  // Reduce from 512 to 320 bits using secp256k1 reduction
  UMultTemplate(t, (r512 + 4), 0x1000003D1ULL);
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);
  UADD1(r512[4], t[4]);

  // Reduce from 320 to 256 bits
  UMultTemplate(t, (r512 + 4), 0x1000003D1ULL);
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  // Final reduction if needed
  if (r512[3] > 0xFFFFFFFEFFFFFC2FULL ||
      (r512[3] == 0xFFFFFFFEFFFFFC2FULL && (r512[2] || r512[1] || r512[0])))
  {
    USUB(r512[0], 0xFFFFFFFEFFFFFC2FULL);
    USUBB(r512[1], 0xFFFFFFFFFFFFFFFFULL);
    USUBB(r512[2], 0xFFFFFFFFFFFFFFFFULL);
    USUBB(r512[3], 0xFFFFFFFFFFFFFFFFULL);
  }

  r[0] = r512[0];
  r[1] = r512[1];
  r[2] = r512[2];
  r[3] = r512[3];
}

// ---------------------------------------------------------------------------------------
// Convenience macros for backward compatibility (avoid UMult redefinition)
// ---------------------------------------------------------------------------------------

// Note: UMult is already defined in GPUMath.h, so we use different names
#define UMultNew(r, a, b) UMultTemplate(r, a, b)
#define ModNeg256_TwoOp(r, a) ModNeg256Template<false>(r, a)
#define ModNeg256_InPlace(r) ModNeg256Template<true>(r)
#define ModSub256_TwoOp(r, a, b) ModSub256Template<false>(r, a, b)
#define ModSub256_InPlace(r, a) ModSub256Template<true>(r, a)
#define ModMult_TwoOp(r, a, b) ModMultTemplate<false>(r, a, b)
#define ModMult_InPlace(r, a) ModMultTemplate<true>(r, a)

#endif // GPUMATHTEMPLATES_H
