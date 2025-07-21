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

#include "Kangaroo.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#endif
#include <cuda_runtime.h>

using namespace std;

#define safe_delete_array(x) if(x) {delete[] x;x=NULL;}

// ----------------------------------------------------------------------------

Kangaroo::Kangaroo(Secp256K1 *secp,int32_t initDPSize,bool useGpu,string &workFile,string &iWorkFile,uint32_t savePeriod,bool saveKangaroo,bool saveKangarooByServer,
                   double maxStep,int wtimeout,int port,int ntimeout,string serverIp,string outputFile,bool splitWorkfile) {

  this->secp = secp;
  this->initDPSize = initDPSize;
  this->useGpu = useGpu;
  this->offsetCount = 0;
  this->offsetTime = 0.0;
  this->workFile = workFile;
  this->saveWorkPeriod = savePeriod;
  this->inputFile = iWorkFile;
  this->nbLoadedWalk = 0;
  this->clientMode = serverIp.length() > 0;
  this->saveKangarooByServer = this->clientMode && saveKangarooByServer;
  this->saveKangaroo = saveKangaroo || this->saveKangarooByServer;
  this->fRead = NULL;
  this->maxStep = maxStep;
  this->wtimeout = wtimeout;
  this->port = port;
  this->ntimeout = ntimeout;
  this->serverIp = serverIp;
  this->outputFile = outputFile;
  this->hostInfo = NULL;
  this->endOfSearch = false;
  this->saveRequest = false;
  this->connectedClient = 0;
  this->totalRW = 0;
  this->collisionInSameHerd = 0;
  this->keyIdx = 0;
  this->splitWorkfile = splitWorkfile;
  this->pid = Timer::getPID();

  CPU_GRP_SIZE = 1024;

  // Init mutex
#ifdef WIN64
  ghMutex = CreateMutex(NULL,FALSE,NULL);
  saveMutex = CreateMutex(NULL,FALSE,NULL);
#else
  pthread_mutex_init(&ghMutex, NULL);
  pthread_mutex_init(&saveMutex, NULL);
  signal(SIGPIPE, SIG_IGN);
#endif

}

// ----------------------------------------------------------------------------

bool Kangaroo::ParseConfigFile(std::string &fileName) {

  // In client mode, config come from the server
  if(clientMode)
    return true;

  // Check file
  FILE *fp = fopen(fileName.c_str(),"rb");
  if(fp == NULL) {
    ::printf("Error: Cannot open %s %s\n",fileName.c_str(),strerror(errno));
    return false;
  }
  fclose(fp);

  // Get lines
  vector<string> lines;
  int nbLine = 0;
  string line;
  ifstream inFile(fileName);
  while(getline(inFile,line)) {

    // Remove ending \r\n
    int l = (int)line.length() - 1;
    while(l >= 0 && isspace(line.at(l))) {
      line.pop_back();
      l--;
    }

    if(line.length() > 0) {
      lines.push_back(line);
      nbLine++;
    }

  }

  if(lines.size()<3) {
    ::printf("Error: %s not enough arguments\n",fileName.c_str());
    return false;
  }

  rangeStart.SetBase16((char *)lines[0].c_str());
  rangeEnd.SetBase16((char *)lines[1].c_str());
  for(int i=2;i<(int)lines.size();i++) {
    
    Point p;
    bool isCompressed;
    if( !secp->ParsePublicKeyHex(lines[i],p,isCompressed) ) {
      ::printf("%s, error line %d: %s\n",fileName.c_str(),i,lines[i].c_str());
      return false;
    }
    keysToSearch.push_back(p);

  }

  ::printf("Start:%s\n",rangeStart.GetBase16().c_str());
  ::printf("Stop :%s\n",rangeEnd.GetBase16().c_str());
  ::printf("Keys :%d\n",(int)keysToSearch.size());

  return true;

}

// ----------------------------------------------------------------------------

bool Kangaroo::IsDP(uint64_t x) {

  return (x & dMask) == 0;

}

void Kangaroo::SetDP(int size) {

  // Mask for distinguised point
  dpSize = size;
  if(dpSize == 0) {
    dMask = 0;
  } else {
    if(dpSize > 64) dpSize = 64;
    dMask = (1ULL << (64 - dpSize)) - 1;
    dMask = ~dMask;
  }

#ifdef WIN64
  ::printf("DP size: %d [0x%016I64X]\n",dpSize,dMask);
#else
  ::printf("DP size: %d [0x%" PRIx64 "]\n",dpSize,dMask);
#endif

}

// ----------------------------------------------------------------------------

bool Kangaroo::Output(Int *pk,char sInfo,int sType) {


  FILE* f = stdout;
  bool needToClose = false;

  if(outputFile.length() > 0) {
    f = fopen(outputFile.c_str(),"a");
    if(f == NULL) {
      printf("Cannot open %s for writing\n",outputFile.c_str());
      f = stdout;
    }
    else {
      needToClose = true;
    }
  }

  if(!needToClose)
    ::printf("\n");

  Point PR = secp->ComputePublicKey(pk);

  ::fprintf(f,"Key#%2d [%d%c]Pub:  0x%s \n",keyIdx,sType,sInfo,secp->GetPublicKeyHex(true,keysToSearch[keyIdx]).c_str());
  if(PR.equals(keysToSearch[keyIdx])) {
    ::fprintf(f,"       Priv: 0x%s \n",pk->GetBase16().c_str());
  } else {
    ::fprintf(f,"       Failed !\n");
    if(needToClose)
      fclose(f);
    return false;
  }


  if(needToClose)
    fclose(f);

  return true;

}

// ----------------------------------------------------------------------------

bool  Kangaroo::CheckKey(Int d1,Int d2,uint8_t type) {

  // Resolve equivalence collision

  if(type & 0x1)
    d1.ModNegK1order();
  if(type & 0x2)
    d2.ModNegK1order();

  Int pk(&d1);
  pk.ModAddK1order(&d2);

  Point P = secp->ComputePublicKey(&pk);

  if(P.equals(keyToSearch)) {
    // Key solved    
#ifdef USE_SYMMETRY
    pk.ModAddK1order(&rangeWidthDiv2);
#endif
    pk.ModAddK1order(&rangeStart);    
    return Output(&pk,'N',type);
  }

  if(P.equals(keyToSearchNeg)) {
    // Key solved
    pk.ModNegK1order();
#ifdef USE_SYMMETRY
    pk.ModAddK1order(&rangeWidthDiv2);
#endif
    pk.ModAddK1order(&rangeStart);
    return Output(&pk,'S',type);
  }

  return false;

}

// ----------------------------------------------------------------------------

bool Kangaroo::CollisionCheck(ENTRY* c1,ENTRY* c2) {

  if(c1->h != c2->h)
    return false;

  Int d1;
  Int d2;
  d1.SetInt32(c1->d);
  d2.SetInt32(c2->d);
  uint32_t h = GetH(c1->x,c1->y);
  if(h!=c1->h) {
    ::printf("[E] GPU/CPU hash mismatch %u %u !\n",h,c1->h);
    return false;
  }

  // To have a collision on distinguished point x and on hash implies a collision in the walk
  // h = sha1(x) & 0xFFFFFFFF
  // Distinguished point is forced using the following condition
  // if( (x & dMask)==0 ) store()

  Point kc1 = secp->ComputePublicKey(&d1);
  Point kc2 = secp->ComputePublicKey(&d2);

  if(kc1.equals(kc2)) {
    // Same position
    return false;
  }

  Int dist;
  dist.Set(&d1);
  dist.Sub(&d2);
  dist.ModK1order();

  Point P = secp->AddDirect(kc1,secp->NegatePoint(kc2));

  if(P.equals(keyToSearch)) {
    // Distinction point
    dist.ModNegK1order();
#ifdef USE_SYMMETRY
    dist.ModAddK1order(&rangeWidthDiv2);
#endif
    dist.ModAddK1order(&rangeStart);
    return Output(&dist,'C',c1->type);
  }

  if(P.equals(keyToSearchNeg)) {
    // Distinction point
#ifdef USE_SYMMETRY
    dist.ModAddK1order(&rangeWidthDiv2);
#endif
    dist.ModAddK1order(&rangeStart);
    return Output(&dist,'C',c1->type);
  }

  return false;

}

// ----------------------------------------------------------------------------

bool Kangaroo::SolveKeyCPU(TH_PARAM *p) {

  bool ok = false;

  // Solve key using CPU

  ENTRY *e1 = &hashtable.Get(p->hIdx);
  if(e1 == NULL) {
    ::printf("SolveKeyCPU: Entry not found\n");
    return false;
  }

  ENTRY e2;
  e2.x = p->x;
  e2.d = p->d.GetInt32();
  e2.h = p->hIdx;
  e2.type = p->type;

  LOCK(ghMutex);
  ok = CollisionCheck(e1,&e2);
  UNLOCK(ghMutex);

  return ok;

}

// ----------------------------------------------------------------------------

bool Kangaroo::SolveKeyGPU(TH_PARAM *p,TH_PARAM * master) {

  bool ok = false;

  // Solve key using GPU

  vector<ENTRY> dps;
  vector<uint32_t> invalidIndex;
  vector<bool> invalid;
  vector<bool> cpu;

  LOCK(ghMutex);
  hashtable.GetCPUStarting(dps,invalidIndex,invalid,cpu);
  UNLOCK(ghMutex);

  if(dps.size() == 0) {
    ::printf("SolveKeyGPU: Table is empty\n");
    return false;
  }

  // Compute lambda key
  GPUEngine g(p->gridSize,p->gpuId,dps.size(),dps,invalidIndex,invalid,cpu);
  g.SetParams(p->dpSize,p->rangePower,p->keysToSearch.size(),p->keyIdx);
  g.Launch(ok,dps);

  if(ok) {
    // Update master
    master->foundKey = true;
    master->key.Set(&p->key);
    master->type = p->type;
  }

  return ok;

}

// ----------------------------------------------------------------------------

bool Kangaroo::SolveKey(TH_PARAM *p) {

  if(p->useGpu)
    return SolveKeyGPU(p,p);
  else
    return SolveKeyCPU(p);

}

// ----------------------------------------------------------------------------

void Kangaroo::InitRange() {

  rangeWidth.Set(&rangeEnd);
  rangeWidth.Sub(&rangeStart);
  rangePower = rangeWidth.GetBitLength();
  ::printf("Range width: 2^%d\n",rangePower);
  rangeWidthDiv2.Set(&rangeWidth);
  rangeWidthDiv2.ShiftR(1);
  rangeWidthDiv4.Set(&rangeWidthDiv2);
  rangeWidthDiv4.ShiftR(1);
  rangeWidthDiv8.Set(&rangeWidthDiv4);
  rangeWidthDiv8.ShiftR(1);

}

void Kangaroo::InitSearchKey() {

  Int SP;
  SP.Set(&rangeStart);
#ifdef USE_SYMMETRY
  SP.ModAddK1order(&rangeWidthDiv2);
#endif
  keyToSearch = secp->ComputePublicKey(&SP);
  keyToSearchNeg = secp->NegatePoint(keyToSearch);

}

// ----------------------------------------------------------------------------

void Kangaroo::CreateJumpTable() {

  double avgDist = pow(2.0,(double)rangePower / 2.0 - 0.5);
  double two_rm1 = pow(2.0,(double)rangePower - 1.0);
  double nuv = log(avgDist) / log(2.0);
  double nuv2 = nuv / 2.0;

  // Choose heuristically optimal parameters mu and k
  // Selected heuristically
  int k = (int)(nuv2 / log(nuv2));
  if (k < 5) k = 5;
  if (rangePower <= 32) k = 5;
  if (k > 128) k = 128;

  jumpBit = k;
  jumpDistance = new Int[k];
  jumpPointx = new Point[k];
  jumpPointy = new Point[k];

  ::printf("Jump avg distance: 2^%.2f\n",log2(avgDist));
  ::printf("Number of jumps: %d\n",k);

  // Select jumps
  Int j;
  Int j2;
  Int maxH = secp->order;
  maxH.ShiftR(rangePower);
  maxH.ShiftR(32);
  if(maxH.IsZero()) maxH.SetInt32(1);
  uint64_t max = maxH.GetInt32();
  double ds = pow(2.0,(double)rangePower / (double)(2 * k));
  j.SetInt32(1);
  for (int i = 0; i < k; i++) {
    jumpDistance[i].Set(&j);
    jumpPointx[i] = secp->ComputePublicKey(&j);
    j.Mult(ds);
    if(j.IsGreater(&maxH)) j.Set(&maxH);
    j2.Rand(&j);
    jumpPointy[i] = secp->ComputePublicKey(&j2);
  }

}

// ----------------------------------------------------------------------------

void Kangaroo::ComputeExpected(double rambda,double &expectedOp,double &expectedMem,double *overHead) {

  // Compute expected operations
  double N = pow(2.0,(double)rangePower);
  double k = (double)jumpBit;
  double w = pow(2.0,(double)rangePower / 2.0 - 0.5);
  double u = 0.42 * sqrt(N);
  expectedOp = 1.05 * u + 10.0 * sqrt((double)nbCPUThread * w);

  // Compute expected memory
  double avgDP0 = u / pow(2.0,(double)dpSize);
  expectedMem = avgDP0 * (HASH_SIZE + 4);

  double op = expectedOp * (1.0 + 0.001 * (double)nbCPUThread * pow(2.0,(double)dpSize) / sqrt(N));
  if(overHead)
    *overHead = op / avgDP0 * 1024.0 * 1024.0;

}

// ----------------------------------------------------------------------------

bool Kangaroo::IsExpected(double op) {

  double u = 0.42 * pow(2.0,(double)rangePower / 2.0);
  double o = 1.05 * u + 10.0 * sqrt((double)nbCPUThread * pow(2.0,(double)rangePower / 2.0 - 0.5));
  return op > o;

}

// ----------------------------------------------------------------------------

void Kangaroo::Run(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize) {

  double t0 = Timer::get_tick();

  nbCPUThread = nbThread;
  if(nbCPUThread<1) nbCPUThread = 1;

  TH_PARAM *params = (TH_PARAM *)malloc(nbCPUThread * sizeof(TH_PARAM));
  THREAD_HANDLE *thHandles = (THREAD_HANDLE *)malloc(nbCPUThread * sizeof(THREAD_HANDLE));

  memset(params,0,nbCPUThread * sizeof(TH_PARAM));

  ::printf("Number of CPU thread: %d\n",nbCPUThread);

  for(int i = 0; i < nbCPUThread; i++) {
    params[i].obj = this;
    params[i].threadId = i;
    params[i].isRunning = true;

#ifdef WIN64
    thHandles[i] = CreateThread(NULL,0,(TH_ENTRY)Kangaroo::Process,(void*)(params + i),0,NULL);
#else
    pthread_create((pthread_t *)&thHandles[i],NULL,(TH_ENTRY)Kangaroo::Process,(void *)(params + i));
#endif
  }

  // Wait that all threads are initialized
  while(!isReady()) {
    Timer::SleepMillis(500);
  }

  double t1 = Timer::get_tick();

  ::printf("Start: %s\n",rangeStart.GetBase16().c_str());
  ::printf("End  : %s\n",rangeEnd.GetBase16().c_str());
  ::printf("Power: 2^%d\n\n",rangePower);

  Process(params,"G",nbCPUThread,gpuId,gridSize);

  // Join
  for(int i = 0; i < nbCPUThread; i++) {
#ifdef WIN64
    WaitForSingleObject(thHandles[i],INFINITE);
#else
    pthread_join((pthread_t)thHandles[i],NULL);
#endif
  }

  free(thHandles);
  free(params);

}

// ----------------------------------------------------------------------------

bool Kangaroo::IsReady() {

  for(int i = 0; i < nbCPUThread; i++) {
    if(!params[i].ready) return false;
  }
  return true;

}

// ----------------------------------------------------------------------------

void Kangaroo::FetchKangaroos(std::vector<ENTRY> &kangaroos) {

  kangaroos.clear();

}

// ----------------------------------------------------------------------------

void Kangaroo::Process(TH_PARAM *params,std::string unit,int nbThread,std::vector<int> gpuId,std::vector<int> gridSize) {

  double t0 = Timer::get_tick();
  uint64_t lastSave = 0;

  while(!endOfSearch) {

    // Compute

    Timer::SleepMillis(1000);

    // Update hash rate
    hashRate = 0.0;
    for(int i = 0; i < nbThread; i++) {
      hashRate += (double)params[i].nbKangaroo;
      params[i].nbKangaroo = 0;
    }

    ::printf("[%.2f Mh/s][GPU %.2f Mh/s][Count 2^%.3f/2^%.3f][Dead %llu][t=%.1fs][%s]\n",
      hashRate / 1000000.0,0.0,(double)totalCount / pow(2.0,30.0),(double)expectedOp / pow(2.0,30.0),collisionInFile,Timer::get_tick() - t0,GetTimeStr(Timer::get_tick() - t0).c_str());

    // Save request
    uint64_t now = (uint64_t)time(NULL);
    if(now - lastSave > saveWorkPeriod) {
      lastSave = now;
      SaveWork();
    }

  }

}

// ----------------------------------------------------------------------------

std::string Kangaroo::GetTimeStr(double s) {

  char timeStr[256];

  if(s < 60.0) {
    sprintf(timeStr,"%.1fs",s);
  } else {
    if(s < 3600.0) {
      sprintf(timeStr,"%2dm %2ds",(int)s / 60,(int)s % 60);
    } else {
      int hour = (int)s / 3600;
      int min = (int)(s - hour * 3600) / 60;
      int sec = (int)(s - hour * 3600 - min * 60);
      sprintf(timeStr,"%dh %02dm %02ds",hour,min,sec);
    }
  }

  return std::string(timeStr);

}

// ----------------------------------------------------------------------------

void Kangaroo::SaveWork() {

  // Save work

}

// ----------------------------------------------------------------------------

bool Kangaroo::LoadWork(std::string &fileName) {

  // Load work
  return true;

}

// ----------------------------------------------------------------------------

void Kangaroo::WorkThread(TH_PARAM *p) {

  // CPU Kangaroo thread

}

// ----------------------------------------------------------------------------

void Kangaroo::DoubleWalk(TH_PARAM *p) {

  // Double walk

}

// ----------------------------------------------------------------------------

void Kangaroo::CreateHerd(int n,Int *px,Int *py,Int *d,int type) {

  // Create Kangaroo herd

}

// ----------------------------------------------------------------------------


