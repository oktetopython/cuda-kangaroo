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
 * along with this program. If not, see <http://www.gnu.org/licenses/>. */

#include "Timer.h"
#include <cstdio>

#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#endif

double Timer::get_tick() {
#ifdef WIN32
  LARGE_INTEGER freq;
  LARGE_INTEGER qpc;
  QueryPerformanceCounter(&qpc);
  QueryPerformanceFrequency(&freq);
  return (double)qpc.QuadPart / (double)freq.QuadPart;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
#endif
}

std::string Timer::getResult(char *title, int nbop, double t0, double t1) {
  char str[256];
  double s = (t1 - t0);
  double ips = (double)nbop / s;
  sprintf(str, "%s %0.2f sec (%2.1f it/s)\n", title, s, ips);
  return std::string(str);
}

void Timer::printResult(char *title, int nbop, double t0, double t1) {
  printf("%s", getResult(title, nbop, t0, t1).c_str());
}