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

#include "Kangaroo.h"
#include "Timer.h"
#include "SECPK1/SECP256k1.h"
#include "GPU/GPUEngine.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>

using namespace std;

#define CHECKARG(opt,n) if(a>=argc-1) {::printf(opt " missing argument #%d\n",n);exit(0);} else {a++;}

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("Kangaroo [-v] [-t nbThread] [-d dpBit] [gpu] [-check]\n");
  printf("         [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y[,g2x,g2y,...]]\n");
  printf("         inFile\n");
  printf(" -v: Print version\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpuId gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y,...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)\n");
  printf(" -d: Specify number of leading zeros for the DP method (default is auto)\n");
  printf(" -t nbThread: Secify number of thread\n");
  printf(" -w workfile: Specify file to save work into (current processed key only)\n");
  printf(" -i workfile: Specify file to load work from (current processed key only)\n");
  printf(" -wi workInterval: Periodic interval (in seconds) for saving work\n");
  printf(" -ws: Save kangaroos in the work file\n");
  printf(" -wss: Save kangaroos via the server\n");
  printf(" -wsplit: Split work file of server and reset hashtable\n");
  printf(" -wm file1 file2 destfile: Merge work file\n");
  printf(" -wmdir dir destfile: Merge directory of work files\n");
  printf(" -wt timeout: Save work timeout in millisec (default is 3000ms)\n");
  printf(" -winfo file1: Work file info file\n");
  printf(" -wpartcreate name: Create empty partitioned work file (name is a directory)\n");
  printf(" -wcheck worfile: Check workfile integrity\n");
  printf(" -m maxStep: number of operations before give up the search (maxStep*expected operation)\n");
  printf(" -s: Start in server mode\n");
  printf(" -c server_ip: Start in client mode and connect to server server_ip\n");
  printf(" -sp port: Server port, default is 17403\n");
  printf(" -nt timeout: Network timeout in millisec (default is 3000ms)\n");
  printf(" -o fileName: output result to fileName\n");
  printf(" -l: List cuda enabled devices\n");
  printf(" -check: Check GPU kernel vs CPU\n");
  printf(" inFile: intput configuration file\n");
  exit(0);

}

// ------------------------------------------------------------------------------------------

int getInt(string name,char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

double getDouble(string name,char *v) {

  double r;

  try {

    r = std::stod(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

// ------------------------------------------------------------------------------------------

void getInts(string name,vector<int> &tokens,const string &text,char sep) {

  size_t start = 0,end = 0;
  tokens.clear();
  int item;

  try {

    while((end = text.find(sep,start)) != string::npos) {
      item = std::stoi(text.substr(start,end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  }
  catch(std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

}
// ------------------------------------------------------------------------------------------

// Default params
static int dp = -1;
static int nbCPUThread;
static string configFile = "";
static bool checkFlag = false;
static bool gpuEnable = false;
static vector<int> gpuId = { 0 };
static vector<int> gridSize;
static string workFile = "";
static string checkWorkFile = "";
static string iWorkFile = "";
static uint32_t savePeriod = 60;
static bool saveKangaroo = false;
static bool saveKangarooByServer = false;
static string merge1 = "";
static string merge2 = "";
static string mergeDest = "";
static string mergeDir = "";
static string infoFile = "";
static double maxStep = 0.0;
static int wtimeout = 3000;
static int ntimeout = 3000;
static int port = 17403;
static bool serverMode = false;
static string serverIP = "";
static string outputFile = "";
static bool splitWorkFile = false;

int main(int argc, char* argv[]) {

  // Global Init
  Timer::Init();
  rseed(Timer::getSeed32());

  ::printf("Kangaroo v" RELEASE "\n");

  int a = 1;

  while (a < argc) {

    if(strcmp(argv[a], "-t") == 0) {
      CHECKARG("-t",1);
      nbCPUThread = atoi(argv[a]);
    } else if (strcmp(argv[a], "-d") == 0) {
      CHECKARG("-d",1);
      dp = atoi(argv[a]);
    } else if (strcmp(argv[a], "-w") == 0) {
      CHECKARG("-w",1);
      workFile = string(argv[a]);
    } else if (strcmp(argv[a], "-i") == 0) {
      CHECKARG("-i",1);
      iWorkFile = string(argv[a]);
    } else if (strcmp(argv[a], "-wi") == 0) {
      CHECKARG("-wi",1);
      savePeriod = atoi(argv[a]);
    } else if (strcmp(argv[a], "-wcheck") == 0) {
      CHECKARG("-wcheck",1);
      checkWorkFile = string(argv[a]);
    } else if (strcmp(argv[a], "-winfo") == 0) {
      CHECKARG("-winfo",1);
      winfo = string(argv[a]);
    } else if (strcmp(argv[a], "-wm") == 0) {
      CHECKARG("-wm",1);
      merge1 = string(argv[a]);
      CHECKARG("-wm",2);
      merge2 = string(argv[a]);
      CHECKARG("-wm",3);
      mergeDest = string(argv[a]);
    } else if (strcmp(argv[a], "-wmdir") == 0) {
      CHECKARG("-wmdir",1);
      mergeDir = string(argv[a]);
      CHECKARG("-wmdir",2);
      mergeDest = string(argv[a]);
    } else if (strcmp(argv[a], "-wt") == 0) {
      CHECKARG("-wt",1);
      saveWorkTimeout = atoi(argv[a]);
    } else if (strcmp(argv[a], "-wpartcreate") == 0) {
      CHECKARG("-wpartcreate",1);
      wpart = string(argv[a]);
    } else if (strcmp(argv[a], "-m") == 0) {
      CHECKARG("-m",1);
      maxStep = atof(argv[a]);
    } else if(strcmp(argv[a], "-gpuId") == 0) {
      CHECKARG("-gpuId",1);
      gpuId.clear();
      string ids = string(argv[a]);
      size_t pos;
      size_t oldPos = 0;
      while((pos = ids.find(",",oldPos)) != string::npos) {
        int id = atoi(ids.substr(oldPos,pos-oldPos).c_str());
        gpuId.push_back(id);
        oldPos = pos + 1;
      }
      int id = atoi(ids.substr(oldPos).c_str());
      gpuId.push_back(id);
    } else if (strcmp(argv[a], "-g") == 0) {
      CHECKARG("-g",1);
      gridSize.clear();
      string dims = string(argv[a]);
      size_t pos;
      size_t oldPos = 0;
      while((pos = dims.find(",",oldPos)) != string::npos) {
        int nb = atoi(dims.substr(oldPos,pos-oldPos).c_str());
        gridSize.push_back(nb);
        oldPos = pos + 1;
      }
      int nb = atoi(dims.substr(oldPos).c_str());
      gridSize.push_back(nb);
    } else if(strcmp(argv[a], "-gpu")==0) {
      gpuEnable = true;
    } else if(strcmp(argv[a], "-check")==0) {
      checkFlag = true;
    } else if(strcmp(argv[a], "-l")==0) {
      GPUEngine::PrintCudaInfo();
      return 0;
    } else if(strcmp(argv[a], "-v")==0) {
      return 0;
    } else if(strcmp(argv[a], "-ws")==0) {
      saveKangaroo = true;
    } else if(strcmp(argv[a], "-wss")==0) {
      saveKangarooByServer = true;
    } else if(strcmp(argv[a], "-s")==0) {
      serverMode = true;
    } else if(strcmp(argv[a], "-c")==0) {
      CHECKARG("-c",1);
      serverIp = string(argv[a]);
    } else if(strcmp(argv[a], "-sp")==0) {
      CHECKARG("-sp",1);
      serverPort = atoi(argv[a]);
    } else if(strcmp(argv[a], "-nt")==0) {
      CHECKARG("-nt",1);
      netTimeout = atoi(argv[a]);
    } else if(strcmp(argv[a], "-o")==0) {
      CHECKARG("-o",1);
      outputFile = string(argv[a]);
    } else if(strcmp(argv[a], "-wsplit")==0) {
      splitWorkfile = true;
    } else {
      if(configFile.length()==0) {
        configFile = string(argv[a]);
      } else {
        printUsage();
      }
    }

    a++;

  }

  if(configFile.length()==0 && checkWorkFile.length()==0 && winfo.length()==0 && merge1.length()==0 && mergeDir.length()==0 && wpart.length()==0) {
    printUsage();
  }

  Kangaroo* wild = new Kangaroo();

  wild->outputFile = outputFile;
  wild->serverIP = serverIp;
  wild->serverPort = serverPort;
  wild->netTimeout = netTimeout;
  wild->saveWorkPeriod = savePeriod;
  wild->saveWorkTimeout = saveWorkTimeout;
  wild->saveKangaroo = saveKangaroo;
  wild->saveKangarooByServer = saveKangarooByServer;
  wild->splitWorkfile = splitWorkfile;

  if(serverMode) {
    wild->RunServer();
    return 0;
  }

  if(checkWorkFile.length()>0) {
    wild->CheckWorkFile(nbCPUThread,checkWorkFile);
    return 0;
  }

  if(winfo.length()>0) {
    wild->WorkInfo(winfo);
    return 0;
  }

  if(merge1.length()>0) {
    wild->MergeWork(merge1,merge2,mergeDest);
    return 0;
  }

  if(mergeDir.length()>0) {
    wild->MergeDir(mergeDir,mergeDest);
    return 0;
  }

  if(wpart.length()>0) {
    wild->CreateEmptyPartWork(wpart);
    return 0;
  }

  if(checkFlag) {
    wild->Check(gpuEnable,gpuId,gridSize);
    return 0;
  }

  if(nbCPUThread==0)
    nbCPUThread = Timer::getCoreNumber();

  if(workFile.length()==0) {
    workFile = "Work_" + configFile.substr(0,configFile.find_last_of(".")) + ".kng";
  }

  if(iWorkFile.length()>0) {
    wild->LoadWork(iWorkFile);
  }

  wild->SetDP(dp);
  wild->isDPauto = (dp<0);

  if(!wild->ParseConfigFile(configFile)) {
    return -1;
  }

  if(maxStep>0) {
    wild->maxStep = maxStep;
  }

  wild->Run(nbCPUThread,gpuEnable,gpuId,gridSize,workFile);

  return 0;

}

if(gpuEnable) {
  k->InitGPU();
}
