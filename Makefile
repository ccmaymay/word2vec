CC := gcc
CXX := g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS := -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS_NO_FUNROLL := -lm -pthread -O3 -march=native -Wall -Wno-unused-result
CFLAGS_NO_MARCH := -lm -pthread -O3 -Wall -funroll-loops -Wno-unused-result
CFLAGS_NO_O3 := -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result
CXXFLAGS := -std=gnu++11 $(CFLAGS)

CUSTOM_WORD2VEC_MAINS := \
	word2vec-no-funroll word2vec-no-march word2vec-no-o3 \
	word2vec-athena-neg word2vec-reservoir-neg word2vec-alias-neg \
	word2vec-athena
WORD2VEC_MAINS := \
	word2vec word2vec-static-window \
	word2vec-unsmoothed-neg word2vec-uniform-neg word2vec-local-vars \
	word2vec-no-subsample word2vec-double word2vec-no-memalign \
	word2vec-no-pthread word2vec-1-neg

SEPARATE_WORD2VEC_RUNTIME_TABS := \
	$(patsubst %,runtime-%.tab,$(WORD2VEC_MAINS) $(CUSTOM_WORD2VEC_MAINS))

SEPARATE_RUNTIME_TABS := \
	$(SEPARATE_WORD2VEC_RUNTIME_TABS) runtime-athena-word2vec.tab runtime-athena-spacesaving-word2vec.tab

NUM_TRIALS ?= 10

.PHONY: all
all: runtime.tab cpuinfo.txt

_math.o: _math.cpp _math.h
	$(CXX) $< -o $@ -c $(CXXFLAGS)

word2vec-athena word2vec-athena-neg word2vec-reservoir-neg word2vec-alias-neg: %: %.c _math.o
	$(CXX) $^ -o $@ $(CXXFLAGS)

word2vec-no-funroll: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_FUNROLL)

word2vec-no-march: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_MARCH)

word2vec-no-o3: word2vec.c
	$(CC) $< -o $@ $(CFLAGS_NO_O3)

word2phrase distance word-analogy compute-accuracy $(WORD2VEC_MAINS): %: %.c
	$(CC) $< -o $@ $(CFLAGS)

text8:
	curl http://mattmahoney.net/dc/text8.zip | gunzip > text8

vocab: word2vec text8
	./word2vec -train text8 -save-vocab vocab

vocab.athena: athena/build/lib/word2vec-vocab-to-naive-lm vocab
	./$^ -s 1e-3 $@

runtime.tab: $(SEPARATE_RUNTIME_TABS)
	sed -s '1!d' $< > $@
	sed -s '1d' $^ >> $@

$(SEPARATE_WORD2VEC_RUNTIME_TABS): runtime-%.tab: % text8 vocab
	./time.bash $(NUM_TRIALS) $@ ./$< -train text8 -read-vocab vocab -output /dev/null -cbow 0 -hs 0 -binary 1 -iter 1 -threads 1

runtime-athena-word2vec.tab: athena/build/lib/word2vec-train-raw text8 vocab.athena
	./time.bash $(NUM_TRIALS) $@ ./$< -e 100 -n 5 -c 5 -t 1e6 -k 0.025 -l vocab.athena text8 /dev/null

runtime-athena-spacesaving-word2vec.tab: athena/build/lib/spacesaving-word2vec-train-raw text8
	./time.bash $(NUM_TRIALS) $@ ./$< -v 1000000 -e 100 -n 5 -c 5 -t 1e6 -k 0.025 text8 /dev/null

cpuinfo.txt:
	cat /proc/cpuinfo > $@

athena/Makefile:
	git clone https://github.com/cjmay/athena

athena/build/lib/spacesaving-word2vec-train-raw: athena/Makefile
	cd athena && make build/lib/spacesaving-word2vec-train-raw

athena/build/lib/word2vec-train-raw: athena/Makefile
	cd athena && make build/lib/word2vec-train-raw

athena/build/lib/word2vec-vocab-to-naive-lm: athena/Makefile
	cd athena && make build/lib/word2vec-vocab-to-naive-lm

clean:
	rm -f $(MAINS) $(SEPARATE_RUNTIME_TABS) runtime.tab cpuinfo.txt vocab
