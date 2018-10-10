#### simple Makefile
#### to compile fast
#### track dependences with -include

src = $(wildcard *.cc)
obj = $(src:.cc=.o)
dep = $(obj:.o=.d)  # one dependency file for each source


#macos = y
#debug = y
#seq = y

CFLAGS =  -DMKL_ILP64 -m64 -I${MKLROOT}/include -I${NUMAROOT}/include -I${TBBROOT}/include -mavx2 -mfma -mf16c -fopenmp -mavx512f -Wall #-march=skylake

ifdef debug
  OPT = -g  -fsanitize=address -O0 -fno-omit-frame-pointer
else
  OPT = -O3 -DNDEBUG
endif

CXX ?= g++
CC = ${CXX} -std=c++11 
#CC = /usr/local/opt/gcc/bin/g++-7 -std=c++11

LDFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl -L${NUMAROOT}/lib -lnuma -L${TBBROOT}/lib -ltbb

mlp_omp: mlp_omp.o Partition.o Rand.o TwistedHyperCube.o
	$(CC) -o $(OPT) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mlp_tbb: mlp_tbb.o Partition.o Rand.o TwistedHyperCube.o
	$(CC) -o $(OPT) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# rule to generate a dep file by using the C preprocessor
# (see man cpp for details on the -MM and -MT options)
%.d: %.cc
	@$(CPP) $(CFLAGS) $< -MM -MT $(@:.d=.o) >$@

%.o: %.cc Makefile
	$(CC) $(CFLAGS) $(OPT) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(obj) ml_perf.exe

.PHONY: cleandep
cleandep:
	rm -f $(dep)

.PHONY: cleanall
cleanall:
	rm -f $(dep) $(obj) ml_perf.exe

