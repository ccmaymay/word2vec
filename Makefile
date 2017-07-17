CC ?= gcc
CXX ?= g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS += -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS_NO_FUNROLL += -lm -pthread -O3 -march=native -Wall -Wno-unused-result
CFLAGS_NO_MARCH += -lm -pthread -O3 -Wall -funroll-loops -Wno-unused-result
CFLAGS_NO_O3 += -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result
CXXFLAGS += -std=gnu++11 $(CFLAGS)

PATCH_TMP ?= patch-tmp

CBLAS_FLAGS ?= -lopenblas

PYTHON ?= python
TRAIN_FILE ?= text8

NUM_TRIALS ?= 10

CUSTOM_WORD2VEC_MAINS := \
	word2vec-no-funroll word2vec-no-march word2vec-no-o3 \
	word2vec-athena-neg word2vec-reservoir-neg word2vec-alias-neg \
	word2vec-athena word2vec-blas word2vec-naive-neg \
	word2vec-blas-alias-neg word2vec-local-vars-more \
	word2vec-local-vars-more-more

WORD2VEC_MAINS := \
	word2vec word2vec-static-window \
	word2vec-unsmoothed-neg word2vec-uniform-neg word2vec-local-vars \
	word2vec-no-subsample word2vec-double word2vec-no-memalign \
	word2vec-no-pthread word2vec-1-neg word2vec-continuous-lr word2vec-exp \
	word2vec-int word2vec-longlong word2vec-comments

SEPARATE_WORD2VEC_RUNTIME_TABS := \
	$(patsubst %,runtime-%.tab,$(WORD2VEC_MAINS) $(CUSTOM_WORD2VEC_MAINS))

SEPARATE_RUNTIME_TABS := \
	$(SEPARATE_WORD2VEC_RUNTIME_TABS) \
	runtime-athena-word2vec.tab runtime-athena-spacesaving-word2vec.tab \
	runtime-gensim-word2vec.tab

.PHONY: all
all: runtime.tab host.txt

_math.o: _math.cpp _math.h
	$(CXX) $< -o $@ -c $(CXXFLAGS)

word2vec-athena word2vec-athena-neg word2vec-reservoir-neg word2vec-alias-neg word2vec-naive-neg: %: %.cpp _math.o
	$(CXX) $^ -o $@ $(CXXFLAGS)

word2vec-blas-alias-neg: word2vec-blas-alias-neg.cpp _math.o
	$(CXX) $^ -o $@ $(CBLAS_FLAGS) $(CXXFLAGS)

word2vec-no-funroll: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_FUNROLL)

word2vec-no-march: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_MARCH)

word2vec-no-o3: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_O3)

word2vec-local-vars-more word2vec-local-vars-more-more: %: %.c
	$(CC) $< -o $@ $(CFLAGS) -std=gnu99

word2vec-blas: word2vec-blas.c
	$(CC) $< -o $@ $(CBLAS_FLAGS) $(CFLAGS)

word2phrase word2vec-effective-tokens distance word-analogy compute-accuracy $(WORD2VEC_MAINS): %: %.c
	$(CC) $< -o $@ $(CFLAGS)

text8:
	curl http://mattmahoney.net/dc/text8.zip | gunzip > text8

text8.split:
	$(PYTHON) split-text8.py text8 text8.split

vocab: word2vec $(TRAIN_FILE)
	./word2vec -train $(TRAIN_FILE) -save-vocab vocab

vocab.athena: athena/build/lib/word2vec-vocab-to-naive-lm vocab
	./$^ -s 1e-3 $@

vocab.gensim: gensim-word2vec.py $(TRAIN_FILE)
	$(PYTHON) $< build-vocab $(TRAIN_FILE) $@

runtime.tab: $(SEPARATE_RUNTIME_TABS)
	sed -s '1!d' $< > $@
	sed -s '1d' $^ >> $@

$(SEPARATE_WORD2VEC_RUNTIME_TABS): runtime-%.tab: % $(TRAIN_FILE) vocab
	./time.bash $(NUM_TRIALS) $@ ./$< -train $(TRAIN_FILE) -read-vocab vocab -output /dev/null -cbow 0 -hs 0 -binary 1 -iter 1 -threads 1

runtime-athena-word2vec.tab: athena/build/lib/word2vec-train-raw $(TRAIN_FILE) vocab.athena
	./time.bash $(NUM_TRIALS) $@ ./$< -e 100 -n 5 -c 5 -t 1e6 -k 0.025 -l vocab.athena $(TRAIN_FILE) /dev/null

runtime-athena-spacesaving-word2vec.tab: athena/build/lib/spacesaving-word2vec-train-raw $(TRAIN_FILE)
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

athena/build/lib/spacesaving-word2vec-train-raw: athena/Makefile
	cd athena && make build/lib/spacesaving-word2vec-train-raw

athena/build/lib/word2vec-train-raw: athena/Makefile
	cd athena && make build/lib/word2vec-train-raw

athena/build/lib/word2vec-vocab-to-naive-lm: athena/Makefile
	cd athena && make build/lib/word2vec-vocab-to-naive-lm

clean:
	rm -f $(WORD2VEC_MAINS)
	rm -f $(CUSTOM_WORD2VEC_MAINS)
	rm -f $(SEPARATE_RUNTIME_TABS) runtime.tab cpuinfo.txt vocab
	rm -rf $(PATCH_TMP)
