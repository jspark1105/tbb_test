#### simple Makefile
#### to compile fast
#### track dependences with -include



src = $(wildcard *.cc)
obj = $(src:.cc=.o)
dep = $(obj:.o=.d)  # one dependency file for each source


#macos = y
#debug = y
#seq = y
SP ?=1
UBN ?=0
TI  ?=0
FG  ?=1
NB  ?=0
CP  ?=1
LT  ?=0
DEFINES = -DSP=${SP} -DPOST_VALIDATION=0 -DUSE_BROADCAST_NODE=$(UBN) -DENABLE_TBB=1 -DTHREAD_INFO=$(TI)
DEFINES+= -DTIME_FG_LOOP=1 -DCOUNT_NODES=1 -DUSE_FG=$(FG) -DNUMA_BIND=$(NB) -DCORE_PINNING=$(CP) -DUSE_LIGHTWEIGHT=$(LT)
CFLAGS =  -DMKL_ILP64 -m64 -I${MKLROOT}/include -I${NUMAROOT}/include -I${TBBROOT}/include -mavx2 -mfma -mf16c -fopenmp -Wall -mavx512f -march=skylake

ifdef debug
  OPT = -g  -fsanitize=address -O0 -fno-omit-frame-pointer
else
  OPT = -O3 -DNDEBUG
endif

#CXX = g++
CXX = icpc
CC = ${CXX} -std=c++11 
LDFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_rt -L${NUMAROOT}/lib -lnuma -L${TBBROOT}/lib -ltbb -L/usr/lib -L/nfs/site/proj/openmp/compilers/intel/19.0/Linux/install/update1/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64 -liomp5

mlp_omp: mlp_omp.o Partition.o Rand.o TwistedHyperCube.o
	$(CC) $(OPT) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mlp_tbb: mlp_tbb.o Partition.o Rand.o TwistedHyperCube.o
	$(CC) $(OPT) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# rule to generate a dep file by using the C preprocessor
# (see man cpp for details on the -MM and -MT options)
%.d: %.cc
	@$(CPP) $(CFLAGS) $< -MM -MT $(@:.d=.o) >$@

%.o: %.cc Makefile
	$(CC) $(CFLAGS) $(DEFINES) $(OPT) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(obj) ml_perf.exe

.PHONY: cleandep
cleandep:
	rm -f $(dep)

.PHONY: cleanall
cleanall:
	rm -f $(dep) $(obj) ml_perf.exe

