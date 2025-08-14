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
#ifdef WITHGPU
#include "GPU/GPUEngine.h"
#endif
#include "CommonUtils.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include <signal.h>
#include <memory>
#include <functional>
#include <atomic>
#ifdef WIN32
#include <io.h> // For _write() function on Windows
#define write _write
#define STDERR_FILENO 2
#else
#include <unistd.h> // For write() function on Unix/Linux
#endif

using namespace std;

#ifdef WITHGPU
// Global RAII GPU memory guard for emergency cleanup
std::unique_ptr<CudaMemoryGuard> g_gpu_guard;
#endif

// Atomic flag for safe signal handling
static std::atomic<bool> g_shutdown_requested{false};
static std::atomic<int> g_received_signal{0};

// Async-signal-safe signal handler
void safe_signal_handler(int signal)
{
  // Only use async-signal-safe functions in signal handlers
  g_received_signal.store(signal);
  g_shutdown_requested.store(true);

  // Use write() instead of printf() - it's async-signal-safe
  const char msg[] = "\nShutdown signal received, cleaning up...\n";
  ssize_t result = write(STDERR_FILENO, msg, sizeof(msg) - 1);
  (void)result; // Suppress unused variable warning
}

// Function to check and handle shutdown signals safely
bool check_and_handle_shutdown()
{
  if (g_shutdown_requested.load())
  {
    int signal = g_received_signal.load();
    printf("Processing shutdown signal %d...\n", signal);

    // Perform safe cleanup
#ifdef WITHGPU
    if (g_gpu_guard)
    {
      printf("Performing GPU cleanup...\n");
      g_gpu_guard.reset(); // This will call the destructor safely
    }
#endif

    printf("Cleanup completed. Exiting...\n");
    return true;
  }
  return false;
}

#define CHECKARG(opt, n)                        \
  if (a >= argc - 1)                            \
  {                                             \
    ::printf(opt " missing argument #%d\n", n); \
    exit(0);                                    \
  }                                             \
  else                                          \
  {                                             \
    a++;                                        \
  }

// Use CommonUtils for unified error handling and constants

// ------------------------------------------------------------------------------------------

void printUsage()
{

  printf("Kangaroo [-v] [-t nbThread] [-d dpBit]");
#ifdef WITHGPU
  printf(" [-gpu] [-check]\n");
  printf("         [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y[,g2x,g2y,...]]\n");
#else
  printf(" [-check]\n");
  printf("         ");
#endif
  printf("         inFile\n");
  printf(" -v: Print version\n");
#ifdef WITHGPU
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpuId gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y,...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)\n");
#endif
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
#ifdef WITHGPU
  printf(" -l: List cuda enabled devices\n");
  printf(" -check: Check GPU kernel vs CPU\n");
#else
  printf(" -check: Check work file integrity\n");
#endif
  printf(" inFile: intput configuration file\n");
  exit(0);
}

// ------------------------------------------------------------------------------------------

// Use CommonUtils functions for parameter parsing

// ------------------------------------------------------------------------------------------

// getInts function moved to CommonUtils

// Argument parsing helper structure
struct ArgHandler
{
  const char *flag;
  int argCount;
  std::function<void(char **)> handler;
};

// Helper function for string argument assignment
void assignStringArg(string &target, char **argv, int &a)
{
  target = string(argv[a]);
  a++;
}

// Helper function for integer argument assignment
void assignIntArg(int &target, const string &name, char **argv, int &a)
{
  target = CommonUtils::getInt(name, argv[a]);
  a++;
}

// Helper function for uint32_t argument assignment
void assignUInt32Arg(uint32_t &target, const string &name, char **argv, int &a)
{
  target = static_cast<uint32_t>(CommonUtils::getInt(name, argv[a]));
  a++;
}

// ------------------------------------------------------------------------------------------

// Default params
static int dp = -1;
static int nbCPUThread;
static string configFile = "";
static bool checkFlag = false;
static bool gpuEnable = false;
#ifdef WITHGPU
static vector<int> gpuId = {0};
static vector<int> gridSize;
#endif
static string workFile = "";
static string checkWorkFile = "";
static string iWorkFile = "";
static uint32_t savePeriod = CommonUtils::Constants::DEFAULT_SAVE_PERIOD;
static bool saveKangaroo = false;
static bool saveKangarooByServer = false;
static string merge1 = "";
static string merge2 = "";
static string mergeDest = "";
static string mergeDir = "";
static string infoFile = "";
static double maxStep = 0.0;
static int wtimeout = CommonUtils::Constants::DEFAULT_TIMEOUT_MS;
static int ntimeout = CommonUtils::Constants::DEFAULT_TIMEOUT_MS;
static int port = CommonUtils::Constants::DEFAULT_SERVER_PORT;
static bool serverMode = false;
static string serverIP = "";
static string outputFile = "";
static bool splitWorkFile = false;

int main(int argc, char *argv[])
{

#ifdef USE_SYMMETRY
  printf("Kangaroo v" RELEASE " (with symmetry)\n");
#else
  printf("Kangaroo v" RELEASE "\n");
#endif

  // Install async-signal-safe signal handlers for emergency GPU cleanup
  signal(SIGINT, safe_signal_handler);
  signal(SIGTERM, safe_signal_handler);
#ifdef WIN32
  signal(SIGBREAK, safe_signal_handler);
#endif

  // Initialize RAII GPU memory guard
#ifdef WITHGPU
  g_gpu_guard = std::make_unique<CudaMemoryGuard>();
#endif

  // Global Init
  Timer::Init();
  rseed(Timer::getSeed32());

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  int a = 1;
  nbCPUThread = Timer::getCoreNumber();

  // Simplified argument parsing to reduce repetitive code

  while (a < argc)
  {
    // Improved argument parsing with reduced repetition
    if (strcmp(argv[a], "-t") == 0)
    {
      CHECKARG("-t", 1);
      assignIntArg(nbCPUThread, "nbCPUThread", argv, a);
    }
    else if (strcmp(argv[a], "-d") == 0)
    {
      CHECKARG("-d", 1);
      assignIntArg(dp, "dpSize", argv, a);
    }
    else if (strcmp(argv[a], "-h") == 0)
    {
      printUsage();
    }
    else if (strcmp(argv[a], "-l") == 0)
    {
#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);
    }
    else if (strcmp(argv[a], "-w") == 0)
    {
      CHECKARG("-w", 1);
      assignStringArg(workFile, argv, a);
    }
    else if (strcmp(argv[a], "-i") == 0)
    {
      CHECKARG("-i", 1);
      assignStringArg(iWorkFile, argv, a);
    }
    else if (strcmp(argv[a], "-wcheck") == 0)
    {
      CHECKARG("-wcheck", 1);
      assignStringArg(checkWorkFile, argv, a);
    }
    else if (strcmp(argv[a], "-winfo") == 0)
    {
      CHECKARG("-winfo", 1);
      assignStringArg(infoFile, argv, a);
    }
    else if (strcmp(argv[a], "-o") == 0)
    {
      CHECKARG("-o", 1);
      assignStringArg(outputFile, argv, a);
    }
    else if (strcmp(argv[a], "-wi") == 0)
    {
      CHECKARG("-wi", 1);
      assignUInt32Arg(savePeriod, "savePeriod", argv, a);
    }
    else if (strcmp(argv[a], "-wt") == 0)
    {
      CHECKARG("-wt", 1);
      assignIntArg(wtimeout, "timeout", argv, a);
    }
    else if (strcmp(argv[a], "-nt") == 0)
    {
      CHECKARG("-nt", 1);
      assignIntArg(ntimeout, "timeout", argv, a);
    }
    else if (strcmp(argv[a], "-m") == 0)
    {
      CHECKARG("-m", 1);
      maxStep = CommonUtils::getDouble("maxStep", argv[a]);
      a++;
    }
    else if (strcmp(argv[a], "-ws") == 0)
    {
      saveKangaroo = true;
      a++;
    }
    else if (strcmp(argv[a], "-wss") == 0)
    {
      saveKangarooByServer = true;
      a++;
    }
    else if (strcmp(argv[a], "-wsplit") == 0)
    {
      splitWorkFile = true;
      a++;
    }
    else if (strcmp(argv[a], "-s") == 0)
    {
      serverMode = true;
      a++;
    }
    else if (strcmp(argv[a], "-c") == 0)
    {
      CHECKARG("-c", 1);
      assignStringArg(serverIP, argv, a);
    }
    else if (strcmp(argv[a], "-sp") == 0)
    {
      CHECKARG("-sp", 1);
      assignIntArg(port, "serverPort", argv, a);
    }
    else if (strcmp(argv[a], "-gpu") == 0)
    {
#ifdef WITHGPU
      gpuEnable = true;
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
      exit(1);
#endif
      a++;
    }
    else if (strcmp(argv[a], "-gpuId") == 0)
    {
#ifdef WITHGPU
      CHECKARG("-gpuId", 1);
      CommonUtils::getInts("gpuId", gpuId, string(argv[a]), ',');
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
      exit(1);
#endif
      a++;
    }
    else if (strcmp(argv[a], "-g") == 0)
    {
#ifdef WITHGPU
      CHECKARG("-g", 1);
      CommonUtils::getInts("gridSize", gridSize, string(argv[a]), ',');
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
      exit(1);
#endif
      a++;
    }
    else if (strcmp(argv[a], "-v") == 0)
    {
      exit(0);
    }
    else if (strcmp(argv[a], "-check") == 0)
    {
      checkFlag = true;
      a++;
    }
    else if (strcmp(argv[a], "-wm") == 0)
    {
      CHECKARG("-wm", 1);
      merge1 = string(argv[a]);
      CHECKARG("-wm", 2);
      merge2 = string(argv[a]);
      a++;
      if (a < argc)
      {
        mergeDest = string(argv[a]);
        a++;
      }
    }
    else if (strcmp(argv[a], "-wmdir") == 0)
    {
      CHECKARG("-wmdir", 1);
      mergeDir = string(argv[a]);
      CHECKARG("-wmdir", 2);
      mergeDest = string(argv[a]);
      a++;
    }
    else if (strcmp(argv[a], "-wpartcreate") == 0)
    {
      CHECKARG("-wpartcreate", 1);
      workFile = string(argv[a]);
      Kangaroo::CreateEmptyPartWork(workFile);
      exit(0);
    }
    else if (a == argc - 1)
    {
      configFile = string(argv[a]);
      a++;
    }
    else
    {
      printf("Unexpected %s argument\n", argv[a]);
      exit(-1);
    }
  }

#ifdef WITHGPU
  if (gridSize.size() == 0)
  {
    for (int i = 0; i < (int)gpuId.size(); i++)
    {
      gridSize.push_back(0);
      gridSize.push_back(0);
    }
  }
  else if (gridSize.size() != gpuId.size() * 2)
  {
    printf("Invalid gridSize or gpuId argument, must have coherent size\n");
    exit(-1);
  }
#endif

  auto v = std::make_unique<Kangaroo>(secp, dp, gpuEnable, workFile, iWorkFile, savePeriod, saveKangaroo, saveKangarooByServer,
                                      maxStep, wtimeout, port, ntimeout, serverIP, outputFile, splitWorkFile);
  if (checkFlag)
  {
#ifdef WITHGPU
    v->Check(gpuId, gridSize);
#else
    v->Check();
#endif
    exit(0);
  }
  else
  {
    if (checkWorkFile.length() > 0)
    {
      v->CheckWorkFile(nbCPUThread, checkWorkFile);
      exit(0);
    }
    if (infoFile.length() > 0)
    {
      v->WorkInfo(infoFile);
      exit(0);
    }
    else if (mergeDir.length() > 0)
    {
      v->MergeDir(mergeDir, mergeDest);
      exit(0);
    }
    else if (merge1.length() > 0)
    {
      v->MergeWork(merge1, merge2, mergeDest);
      exit(0);
    }
    if (iWorkFile.length() > 0)
    {
      if (!v->LoadWork(iWorkFile))
        exit(-1);
    }
    else if (configFile.length() > 0)
    {
      if (!v->ParseConfigFile(configFile))
        exit(-1);
    }
    else
    {
      if (serverIP.length() == 0)
      {
        ::printf("No input file to process\n");
        exit(-1);
      }
    }
  }

  // Run the main algorithm with exception handling and signal checking
  try
  {
    // Check for shutdown signals before starting main algorithm
    if (check_and_handle_shutdown())
    {
      return 0;
    }

    if (serverMode)
      v->RunServer();
    else
#ifdef WITHGPU
      v->Run(nbCPUThread, gpuId, gridSize);
#else
      v->Run(nbCPUThread);
#endif

    // Check for shutdown signals after main algorithm completes
    if (check_and_handle_shutdown())
    {
      return 0;
    }
  }
  catch (const std::exception &e)
  {
    printf("Exception caught: %s\n", e.what());

    // Check if this was caused by a signal
    if (check_and_handle_shutdown())
    {
      delete secp;
      return 0; // Clean shutdown due to signal
    }

#ifdef WITHGPU
    GPUEngine::ForceGPUCleanup();
#endif
    delete secp;
    // v automatically cleaned by unique_ptr destructor
    return -1;
  }
  catch (...)
  {
    printf("Unknown exception caught\n");

    // Check if this was caused by a signal
    if (check_and_handle_shutdown())
    {
      delete secp;
      return 0; // Clean shutdown due to signal
    }

#ifdef WITHGPU
    GPUEngine::ForceGPUCleanup();
#endif
    delete secp;
    // v automatically cleaned by unique_ptr destructor
    return -1;
  }

  // Clean shutdown - RAII will automatically clean up
  delete secp;
  // v automatically cleaned by unique_ptr destructor

  // GPU memory will be automatically cleaned by RAII guard destructor
#ifdef WITHGPU
  printf("Program completed successfully - GPU memory cleaned\n");
#else
  printf("Program completed successfully\n");
#endif

  return 0;
}
