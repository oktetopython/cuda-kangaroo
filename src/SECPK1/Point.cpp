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
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "Point.h"
#include "Int.h"

Point::Point() {
}

Point::Point(const Point &p) {
  x = p.x;
  y = p.y;
  z = p.z;
}

void Point::Set(const Point &p) {
  x = p.x;
  y = p.y;
  z = p.z;
}

void Point::Set(Point &p) {
  x = p.x;
  y = p.y;
  z = p.z;
}

void Point::Clear() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

bool Point::isZero() {
  return x.IsZero() && y.IsZero();
}

void Point::Reduce() {
  Int iv(&z);
  iv.ModInv();
  x.ModMul(&x,&iv);
  y.ModMul(&y,&iv);
  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  Int pz2(p.z);
  pz2.ModSquareK1(&p.z);

  Int t(x);
  t.ModMulK1(&t,&pz2);

  Int pz3(pz2);
  pz3.ModMulK1(&pz3,&p.z);

  Int q(y);
  q.ModMulK1(&q,&pz3);

  Int r1(p.x);
  r1.ModMulK1(&r1,&z);
  r1.ModMulK1(&r1,&z);

  Int r2(p.y);
  r2.ModMulK1(&r2,&z);
  r2.ModMulK1(&r2,&z);
  r2.ModMulK1(&r2,&z);

  if(!t.Equals(&r1)) return false;
  if(!q.Equals(&r2)) return false;

  return true;
}

std::string Point::toString() {
  std::string ret = "G(";
  ret += x.GetBase16();
  ret += " , ";
  ret += y.GetBase16();
  ret += " , ";
  ret += z.GetBase16();
  ret += ")";
  return ret;
}

void Point::print() {
  printf("%s\n",toString().c_str());
}