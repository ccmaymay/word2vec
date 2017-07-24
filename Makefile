CC ?= gcc
CXX ?= g++
BASE_CFLAGS ?= -lm -pthread -Wall -Wno-unused-result
CFLAGS += $(BASE_CFLAGS)
CFLAGS_NO_FUNROLL += $(BASE_CFLAGS)
CFLAGS_NO_MARCH += $(BASE_CFLAGS)
CFLAGS_NO_O3 += $(BASE_CFLAGS)
CXXFLAGS += -std=gnu++11 $(BASE_CFLAGS)

ifdef DEBUG
	CFLAGS += -O0 -g3 -gdwarf-2
	CFLAGS_NO_FUNROLL += -O0 -g3 -gdwarf-2
	CFLAGS_NO_MARCH += -O0 -g3 -gdwarf-2
	CFLAGS_NO_O3 += -O0 -g3 -gdwarf-2
	CXXFLAGS += -O0 -g3 -gdwarf-2
else
	CFLAGS += -O3 -march=native -funroll-loops
	CFLAGS_NO_FUNROLL += -O3 -march=native
	CFLAGS_NO_MARCH += -O3 -funroll-loops
	CFLAGS_NO_O3 += -O2 -march=native -funroll-loops
	CXXFLAGS += -O3 -march=native -funroll-loops
endif

QUERY ?= bush

CBLAS_FLAGS ?= -lopenblas

PYTHON ?= python
TRAIN_FILE ?= text8

NUM_TRIALS ?= 10

LIBATHENA_SOURCES := $(wildcard athena/src/_*.cpp)
LIBATHENA_HEADERS := $(wildcard athena/src/_*.h)

WORD2VEC_ATHENA_MAINS := \
	word2vec-athena-0 word2vec-athena-1 word2vec-athena-2 word2vec-athena-3 \
	word2vec-athena-4 word2vec-athena-5 word2vec-athena-6 word2vec-athena-7 \
	word2vec-athena-8 word2vec-athena-9 word2vec-athena-10 word2vec-athena-11 \
	word2vec-athena-12 word2vec-athena-13 word2vec-athena-14 \
	word2vec-athena-15

WORD2VEC_ATHENA_NEG_MAINS := \
	word2vec-athena-neg word2vec-reservoir-neg word2vec-alias-neg \
    word2vec-naive-neg

CUSTOM_WORD2VEC_MAINS := \
	$(WORD2VEC_ATHENA_MAINS) \
	$(WORD2VEC_ATHENA_NEG_MAINS) \
	word2vec-no-funroll word2vec-no-march word2vec-no-o3 \
	word2vec-blas word2vec-blas-alias-neg word2vec-local-vars-1 \
	word2vec-local-vars-2

WORD2VEC_MAINS := \
	word2vec word2vec-static-window \
	word2vec-unsmoothed-neg word2vec-uniform-neg word2vec-local-vars-0 \
	word2vec-no-subsample word2vec-double word2vec-no-memalign \
	word2vec-no-pthread word2vec-1-neg word2vec-continuous-lr word2vec-exp \
	word2vec-int word2vec-longlong word2vec-comments

SEPARATE_WORD2VEC_RUNTIME_TABS := \
	$(patsubst %,runtime-%.tab,$(WORD2VEC_MAINS) $(CUSTOM_WORD2VEC_MAINS))

SEPARATE_WORD2VEC_MODELS := \
	$(patsubst %,model-%.bin,$(WORD2VEC_MAINS) $(CUSTOM_WORD2VEC_MAINS))

SEPARATE_WORD2VEC_QUERY_OUTPUTS := \
	$(patsubst %,query-%.txt,$(WORD2VEC_MAINS) $(CUSTOM_WORD2VEC_MAINS))

SEPARATE_RUNTIME_TABS := \
	$(SEPARATE_WORD2VEC_RUNTIME_TABS) \
	runtime-athena-word2vec.tab runtime-athena-spacesaving-word2vec.tab \
	runtime-gensim-word2vec.tab

.PHONY: all
all: runtime.tab host.txt

_math.o: _math.cpp _math.h
	$(CXX) $< -o $@ -c $(CXXFLAGS)

$(WORD2VEC_ATHENA_NEG_MAINS): %: %.cpp _math.o
	$(CXX) $^ -o $@ $(CXXFLAGS)

word2vec-athena-3-num-pairs word2vec-athena-4-num-pairs: %: %.cpp libathena.a
	$(CXX) $^ -o $@ $(CXXFLAGS) -fopenmp

$(WORD2VEC_ATHENA_MAINS): %: %.cpp libathena.a
	$(CXX) $^ -o $@ $(CXXFLAGS) -fopenmp

word2vec-blas-alias-neg: word2vec-blas-alias-neg.cpp _math.o
	$(CXX) $^ -o $@ $(CBLAS_FLAGS) $(CXXFLAGS)

word2vec-no-funroll: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_FUNROLL)

word2vec-no-march: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_MARCH)

word2vec-no-o3: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_O3)

word2vec-local-vars-1 word2vec-local-vars-2: %: %.c
	$(CC) $< -o $@ $(CFLAGS) -std=gnu99

word2vec-blas: word2vec-blas.c
	$(CC) $< -o $@ $(CBLAS_FLAGS) $(CFLAGS)

word2vec-num-pairs word2vec-entropy word2phrase word2vec-effective-tokens distance word-analogy compute-accuracy $(WORD2VEC_MAINS): %: %.c
	$(CC) $< -o $@ $(CFLAGS)

text8:
	curl http://mattmahoney.net/dc/text8.zip | gunzip > text8

text8.split:
	$(PYTHON) split-text8.py text8 text8.split

vocab: word2vec $(TRAIN_FILE)
	./word2vec -train $(TRAIN_FILE) -save-vocab vocab

vocab.athena: word2vec-vocab-to-naive-lm vocab
	./$^ -s 1e-3 $@

vocab.gensim: gensim-word2vec.py $(TRAIN_FILE)
	$(PYTHON) $< build-vocab $(TRAIN_FILE) $@

runtime.tab: $(SEPARATE_RUNTIME_TABS)
	sed -s '1!d' $< > $@
	sed -s '1d' $^ >> $@

$(SEPARATE_WORD2VEC_RUNTIME_TABS): runtime-%.tab: % $(TRAIN_FILE) vocab
	./time.bash $(NUM_TRIALS) $@ ./$< -train $(TRAIN_FILE) -read-vocab vocab -output /dev/null -cbow 0 -hs 0 -binary 1 -iter 1 -threads 1

$(SEPARATE_WORD2VEC_MODELS): model-%.bin: % $(TRAIN_FILE) vocab
	./$< -train $(TRAIN_FILE) -read-vocab vocab -output $@ -cbow 0 -hs 0 -binary 1 -iter 1 -threads 1

$(SEPARATE_WORD2VEC_QUERY_OUTPUTS): query-%.txt: model-%.bin distance
	echo $(QUERY) | ./distance $< | sed '/^Enter word/d;1,/^--------/d' | awk '{ print $$1 }' > $@

query.txt: $(SEPARATE_WORD2VEC_QUERY_OUTPUTS)
	head $^ > $@

runtime-athena-word2vec.tab: word2vec-train $(TRAIN_FILE) vocab.athena
	./time.bash $(NUM_TRIALS) $@ ./$< -e 100 -n 5 -c 5 -k 0.025 -l vocab.athena $(TRAIN_FILE) /dev/null

runtime-athena-word2vec-alias.tab: word2vec-alias-train $(TRAIN_FILE) vocab.athena
	./time.bash $(NUM_TRIALS) $@ ./$< -e 100 -n 5 -c 5 -k 0.025 -l vocab.athena $(TRAIN_FILE) /dev/null

runtime-athena-word2vec-blas-alias.tab: word2vec-blas-alias-train $(TRAIN_FILE) vocab.athena
	./time.bash $(NUM_TRIALS) $@ ./$< -e 100 -n 5 -c 5 -k 0.025 -l vocab.athena $(TRAIN_FILE) /dev/null

runtime-athena-spacesaving-word2vec.tab: spacesaving-word2vec-train $(TRAIN_FILE)
	./time.bash $(NUM_TRIALS) $@ ./$< -v 1000000 -e 100 -n 5 -c 5 -t 1e6 -k 0.025 $(TRAIN_FILE) /dev/null

runtime-gensim-word2vec.tab: gensim-word2vec.py $(TRAIN_FILE) vocab.gensim
	./time.bash $(NUM_TRIALS) $@ $(PYTHON) $< train-model $(TRAIN_FILE) vocab.gensim /dev/null

host.txt:
	hostname > $@
	echo >> $@; cat /proc/cpuinfo >> $@
	echo >> $@; free -m >> $@
	echo >> $@; $(CC) --version >> $@ 2>&1
	echo >> $@; $(CXX) --version >> $@ 2>&1
	echo >> $@; $(PYTHON) --version >> $@ 2>&1
	echo >> $@; $(PYTHON) -c 'import gensim; print("gensim " + gensim.__version__)' >> $@

athena/Makefile:
	git clone https://github.com/cjmay/athena

spacesaving-word2vec-train: athena/Makefile $(LIBATHENA_SOURCES) $(LIBATHENA_HEADERS) athena/src/spacesaving-word2vec-train.cpp
	cd athena && make clean build/bin/$@ && cp build/bin/$@ ../$@

word2vec-train: athena/Makefile $(LIBATHENA_SOURCES) $(LIBATHENA_HEADERS) athena/src/word2vec-train.cpp
	cd athena && make clean build/bin/$@ && cp build/bin/$@ ../$@

word2vec-alias-train: athena/Makefile $(LIBATHENA_SOURCES) $(LIBATHENA_HEADERS) athena/src/word2vec-alias-train.cpp
	cd athena && make clean build/bin/$@ && cp build/bin/$@ ../$@

word2vec-blas-alias-train: athena/Makefile $(LIBATHENA_SOURCES) $(LIBATHENA_HEADERS) athena/src/word2vec-alias-train.cpp
	cd athena && make clean build/bin/word2vec-alias-train HAVE_CBLAS=1 && cp build/bin/word2vec-alias-train ../$@

word2vec-vocab-to-naive-lm: athena/Makefile $(LIBATHENA_SOURCES) $(LIBATHENA_HEADERS) athena/src/word2vec-vocab-to-naive-lm.cpp
	cd athena && make clean build/bin/$@ && cp build/bin/$@ ../$@

libathena.a: $(LIBATHENA_SOURCES) $(LIBATHENA_HEADERS)
	cd athena && make clean build/lib/$@ && cp build/lib/$@ ../$@

clean:
	rm -f $(WORD2VEC_MAINS)
	rm -f $(CUSTOM_WORD2VEC_MAINS)
	rm -f $(SEPARATE_RUNTIME_TABS) runtime.tab cpuinfo.txt vocab
