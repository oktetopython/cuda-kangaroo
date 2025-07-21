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
 * along with this program. If not, see <http://www.gnu.org/licenses/>. */

#include "IntGroup.h"

IntGroup::IntGroup(int size) {
  this->size = size;
  ints = new Int[size];
  subp = new Int[size];
}

IntGroup::~IntGroup() {
  delete[] ints;
  delete[] subp;
}

void IntGroup::Set(Int *pts) {
  for (int i = 0; i < size; i++)
    ints[i].Set(pts[i]);
}

void IntGroup::ModInv() {

  Int newValue;
  Int accumulator;

  subp[0].Set(&ints[0]);
  for (int i = 1; i < size; ++i) {
    subp[i].Set(&ints[i]);
    for (int j = 0; j < i; ++j) {
      accumulator.ModMulK1(&subp[j], &ints[i]);
      subp[i].ModSub(&accumulator, &subp[i]);
    }
  }

  accumulator.ModInv(&subp[size - 1]);

  for (int i = size - 2; i > 0; --i) {
    newValue.ModMulK1(&subp[i], &accumulator);
    accumulator.ModMulK1(&newValue, &ints[i + 1]);
  }

  accumulator.ModMulK1(&subp[0], &accumulator);

  for (int i = 1; i < size; ++i) {
    newValue.Set(&accumulator);
    for (int j = 0; j < i; ++j) {
      accumulator.ModMulK1(&subp[j], &ints[i]);
      newValue.ModSub(&accumulator, &newValue);
    }
    ints[i].ModMulK1(&newValue, &subp[i]);
  }

  ints[0].Set(&accumulator);

}