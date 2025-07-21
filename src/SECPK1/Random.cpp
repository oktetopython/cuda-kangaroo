/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
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
 * along with this program. If not, see <http://www.gnu.org/licenses/>. */

#include "Random.h"
#include "Timer.h"
#include <string.h>

#if defined(_WIN64)
#include <windows.h>
#include <wincrypt.h>
#endif

#define SAFE_FREE(p) { if(p) {free(p); p=NULL;} }

static uint32_t _seed[32];

void _RandomSeed() {

  // Also use Secp256k1 rand seed...

#if defined(_WIN64)
  HCRYPTPROV hProv;
  if(CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
    DWORD len = sizeof(_seed);
    CryptGenRandom(hProv, len, (BYTE *)_seed);
    CryptReleaseContext(hProv, 0);
  }
#else
  FILE *f = fopen("/dev/urandom", "r");
  if (f) {
    fread(_seed, 1, sizeof(_seed), f);
    fclose(f);
  }
#endif

  // "Improve entropy"
  double start = Timer::get_tick();
  uint64_t i = (uint64_t)start;
  for (i = 0; i < 100000; i++);
  double stop = Timer::get_tick();
  _seed[0] ^= (uint32_t)(uint64_t)start;
  _seed[1] ^= (uint32_t)((uint64_t)start >> 32);
  _seed[2] ^= (uint32_t)(uint64_t)stop;
  _seed[3] ^= (uint32_t)((uint64_t)stop >> 32);
  _seed[4] ^= (uint32_t)(uint64_t)i;
  _seed[5] ^= (uint32_t)(i >> 32);

}

void InitSeed() {
  _RandomSeed();
}

uint64_t rand64() {

  // Well equidistributed long-period linear PRNG https://en.wikipedia.org/wiki/Xorshift

  uint64_t x = *(uint64_t *)(_seed);
  x ^= x >> 12; // a
  x ^= x << 25; // b
  x ^= x >> 27; // c
  *(uint64_t *)(_seed) = x;
  return x * UINT64_C(2685821657736338717);

}

uint32_t rand32() {
  return (uint32_t)rand64();
}