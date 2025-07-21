/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef POINTH
#define POINTH

#include "Int.h"

class Point {

public:

  Point();
  Point(const Point &p);
  Point& operator=(const Point &p);
  bool operator==(const Point &p) const;
  bool operator!=(const Point &p) const;

  void Set(const Point &p);
  void Set(Int *cx, Int *cy);
  void Clear();
  bool isZero() const;

  void Reduce();

  Int x;
  Int y;
  bool valid;

};

#endif // POINTH