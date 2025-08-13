#---------------------------------------------------------------------
# Makefile for BSGS
#
# Author : Jean-Luc PONS

ifdef gpu

SRC = SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      GPU/GPUEngine.o GPU/GeccGPUWrapper.o Kangaroo.cpp HashTable.cpp \
      Backup.cpp Thread.cpp Check.cpp Network.cpp Merge.cpp PartMerge.cpp \
      SECPK1/GeccAdapter.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      GPU/GPUEngine.o GPU/GeccGPUWrapper.o Kangaroo.o HashTable.o Thread.o \
      Backup.o Check.o Network.o Merge.o PartMerge.o \
      SECPK1/GeccAdapter.o)

else

SRC = SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      Kangaroo.cpp HashTable.cpp Thread.cpp Check.cpp \
      Backup.cpp Network.cpp Merge.cpp PartMerge.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      Kangaroo.o HashTable.o Thread.o Check.o Backup.o \
      Network.o Merge.o PartMerge.o)

endif

CXX        = g++
# Auto-detect CUDA installation
CUDA       = $(shell if [ -d "/usr/local/cuda" ]; then echo "/usr/local/cuda"; elif [ -d "/usr/local/cuda-12.0" ]; then echo "/usr/local/cuda-12.0"; elif [ -d "/usr/local/cuda-11.0" ]; then echo "/usr/local/cuda-11.0"; else echo "/usr/local/cuda-8.0"; fi)
CXXCUDA    = g++
NVCC       = $(CUDA)/bin/nvcc

ifdef gpu

ifdef debug
CXXFLAGS   = -DWITHGPU -m64  -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I$(CUDA)/include -I./gECC/include -I./gECC/build/include
else
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include -I./gECC/include -I./gECC/build/include
endif
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart -L./gECC/build/lib -lgecc

else

ifdef debug
CXXFLAGS   = -m64 -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I./gECC/include -I./gECC/build/include
else
CXXFLAGS   =  -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I./gECC/include -I./gECC/build/include
endif
LFLAGS     = -lpthread -L./gECC/build/lib -lgecc

endif

#--------------------------------------------------------------------

ifdef gpu
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu

$(OBJDIR)/GPU/GeccGPUWrapper.o: GPU/GeccGPUWrapper.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include -I./gECC/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GeccGPUWrapper.o -c GPU/GeccGPUWrapper.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu

$(OBJDIR)/GPU/GeccGPUWrapper.o: GPU/GeccGPUWrapper.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -I./gECC/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GeccGPUWrapper.o -c GPU/GeccGPUWrapper.cu
endif
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: bsgs

bsgs: gecc_lib $(OBJET)
	@echo Making Kangaroo...
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo

gecc_lib:
	@echo Making gECC library...
	@cd gECC && mkdir -p build && cd build && cmake .. && make

$(OBJET): | $(OBJDIR) $(OBJDIR)/SECPK1 $(OBJDIR)/GPU

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

$(OBJDIR)/SECPK1: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p SECPK1

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/SECPK1/*.o

