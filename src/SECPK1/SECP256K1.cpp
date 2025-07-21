/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "SECP256k1.h"
#include "hash/sha256.h"
#include "hash/ripemd160.h"
#include "Base58.h"
#include "Bech32.h"
#include <string.h>

Secp256K1::Secp256K1() {
}

void Secp256K1::Init() {
  // Prime for the finite field
  Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order
  G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  Int::InitK1(&order);

  // Compute Generator table
  Point N(G);
  for(int i = 0; i < 32; i++) {
    GTable[i * 256] = N;
    N = DoubleDirect(N);
    for (int j = 1; j < 255; j++) {
      GTable[i * 256 + j] = N;
      N = AddDirect(N, GTable[i * 256]);
    }
    GTable[i * 256 + 255] = N; // Dummy point for check function
  }
}

Secp256K1::~Secp256K1() {
}

void PrintResult(bool ok) {
  if(ok) {
    printf("OK\n");
  }
  else {
    printf("Failed !\n");
  }
}

void CheckAddress(Secp256K1 *T,std::string address,std::string privKeyStr) {
  Int privKey = T->DecodePrivateKey((char *)privKeyStr.c_str(),NULL);
  Point p = T->ComputePublicKey(&privKey);
  std::string calcAddress = T->GetAddress(P2PKH,true,p);
  bool ok = (calcAddress == address);
  printf("Address : %s\n",calcAddress.c_str());
  PrintResult(ok);
}

void Secp256K1::Check() {

  // Add/Sub/Mul/Div/Inv test
  Int a,b,c,d,e;

  a.SetBase16("C84A53D18D9D6C8F9BB791E153C83785DF6E106D6A423A25E4B796A9DF228C03");
  b.SetBase16("2DFFE21E71B44D653F36CA8D隆