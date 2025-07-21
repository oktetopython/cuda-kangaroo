/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/VanitySearch).
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

// Big integer class (Fixed size)

#ifndef BIGINTH
#define BIGINTH

#include "Random.h"
#include <string>
#include <inttypes.h>

// We need 1 extra block for Knuth div algorithm , Montgomery multiplication and ModInv
#define BISIZE 256

#if BISIZE==256
  #define NB64BLOCK 5
  #define NB32BLOCK 10
#elif BISIZE==512
  #define NB64BLOCK 9
  #define NB32BLOCK 18
#else
  #error Unsuported size
#endif

class Int {

public:

  Int();
  Int(int64_t i64);
  Int(uint64_t u64);
  Int(Int *a);

  // Op
  void Add(uint64_t a);
  void Add(Int *a);
  void Add(Int *a,Int *b);
  void AddOne();
  void Sub(uint64_t a);
  void Sub(Int *a);
  void Sub(Int *a, Int *b);
  void SubOne();
  void Mult(Int *a);
  uint64_t Mult(uint64_t a);
  uint64_t IMult(int64_t a);
  uint64_t Mult(Int *a,uint64_t b);
  uint64_t IMult(Int *a, int64_t b);
  void Mult(Int *a,Int *b);
  void Div(Int *a,Int *mod = NULL);
  void MultModN(Int *a, Int *b, Int *n);
  void Neg();
  void Abs();

  // Right shift (to follow MSB 0x80000000)
  void SHIFT64R();
  void SHIFT32R();
  void RShift(int n,bool modulo = false);
  void LShift(int n);
  void RShiftD();
  void SwapBit(int bitNumber);
  int GetBit(int n);

  // Comp
  int Compare(Int *a);
  int CompareToZero();
  bool IsGreater(Int *a);
  bool IsLower(Int *a);
  bool IsGreaterOrEqual(Int *a);
  bool IsLowerOrEqual(Int *a);
  bool IsEqual(Int *a);
  bool IsNotEqual(Int *a);
  bool IsZero();
  bool IsOne();
  bool IsStrictPositive();
  bool IsPositive();
  bool IsNegative();
  bool IsEven();
  bool IsOdd();

  // Modular
  void GCD(Int *a);
  void Mod(Int *n);
  void ModInv(Int *n);
  void ModAdd(Int *a,Int *mod);
  void ModSub(Int *a,Int *mod);
  void ModMul(Int *a,Int *mod);
  void ModMul(Int *a,Int *b,Int *mod);
  void ModExp(Int *e,Int *mod);
  void ModNeg(Int *mod);
  void ModSqrt(Int *mod);
  bool IsSqrt();
  Int* GetFieldCharacteristic();
  Int* GetR();
  Int* GetR2();
  Int* GetR3();
  Int* GetR4();
  static int GetModMultSize();
  static uint64_t GetMM64();
  static uint32_t GetMM32();

  // Size
  int GetSize();
  int GetSizeBits();
  int GetBitLength();

  // Setter
  void SetInt32(uint32_t value);
  void SetInt64(uint64_t value);
  void Set(Int *a);
  void SetBase10(char *value);
  void SetBase16(char *value);
  void SetBaseN(int n,char *charset,char *value);
  void SetByte(int bytePos,uint8_t byte);
  void MaskByte(int n);

  // Getter
  uint32_t GetInt32();
  uint32_t GetInt32(int n);
  uint8_t GetByte(int bytePos);
  void  Get32Bytes(unsigned char *buff);
  char* GetBase10();
  char* GetBase16();
  char* GetCStr();
  char* GetBaseN(int n,char *charset);
  uint64_t Get64(int idx);
  uint32_t Get32(int idx);

  // Check functions
  static void Check();

  // Align macro
  void ALIGN();

  uint64_t bits64[NB64BLOCK]; // Number
  char vstr[256];

};

#endif // BIGINTH