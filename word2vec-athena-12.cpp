//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

#include <athena/src/_math.h>
#include <athena/src/_core.h>
#include <athena/src/_cblas.h>

#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

using namespace std;

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
int binary = 0, debug_mode = 2, min_count = 5, num_threads = 12, min_reduce = 1;
long long word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
clock_t start;

int negative = 5;
const int table_size = 1e8;

void zero_vector(long long n, real* v) {
  memset(v, 0, n * sizeof(real));
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof, char *eos) {
  int a = 0;
  while (1) {
    int ch = fgetc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        *eos = 1;
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(const NaiveLanguageModel& language_model, FILE *fin, char *eof, char *eos) {
  char word[MAX_STRING], eof_l = 0, eos_l = 0;
  ReadWord(word, fin, &eof_l, &eos_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  if (eos_l) {
    *eos = 1;
    return -1;
  }
  return language_model.lookup(word);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(NaiveLanguageModel& language_model) {
  long new_lm_size = 0;
  for (int a = 0; a < language_model.size(); a++) if (language_model.count(a) >= min_count) {
    ++new_lm_size;
  }
  language_model.truncate(new_lm_size);
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(NaiveLanguageModel& language_model) {
  long new_lm_size = 0;
  for (int a = 0; a < language_model.size(); a++) if (language_model.count(a) > min_reduce) {
    ++new_lm_size;
  }
  language_model.truncate(new_lm_size);
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile(NaiveLanguageModel& language_model) {
  long long wc = 0;
  FILE *fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  while (1) {
    char word[MAX_STRING];
    char eof = 0, eos = 0;
    ReadWord(word, fin, &eof, &eos);
    if (eof) break;
    if (eos) continue;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%zuM%c", language_model.total() / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    language_model.increment(word);
    if (language_model.size() > vocab_hash_size * 0.7) ReduceVocab(language_model);
  }
  SortVocab(language_model);
  if (debug_mode > 0) {
    printf("Vocab size: %zu\n", language_model.size());
    printf("Words in train file: %zu\n", language_model.total());
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab(const NaiveLanguageModel& language_model) {
  FILE *fo = fopen(save_vocab_file, "wb");
  for (long long i = 0; i < language_model.size(); i++) fprintf(fo, "%s %zu\n", language_model.reverse_lookup(i).c_str(), language_model.count(i));
  fclose(fo);
}

void ReadVocab(NaiveLanguageModel& language_model) {
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  while (1) {
    char word[MAX_STRING];
    char eof = 0, eos = 0;
    ReadWord(word, fin, &eof, &eos);
    if (eof) break;
    if (eos) continue;
    char c;
    long long count = 0;
    fscanf(fin, "%lld%c", &count, &c);
    for (long long j = 0; j < count; ++j) {
      language_model.increment(word);
    }
  }
  SortVocab(language_model);
  if (debug_mode > 0) {
    printf("Vocab size: %zu\n", language_model.size());
    printf("Words in train file: %zu\n", language_model.total());
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void update_progress(const NaiveLanguageModel& language_model, const SGD& sgd) {
  if ((debug_mode > 1)) {
    clock_t now = clock();
    printf("%cAlpha[0]: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, sgd.get_rho(0),
     word_count_actual / (real)(iter * language_model.total() + 1) * 100,
     word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
    fflush(stdout);
  }
}

long long read_new_sentence(FILE* fi, const NaiveLanguageModel& language_model,
                            long long* word_count, long long* sen, char* eof) {
  long long sentence_length = 0;
  uniform_real_distribution<float> d(0, 1);
  while (1) {
    char eos = 0;
    long long word = ReadWordIndex(language_model, fi, eof, &eos);
    if (*eof) break;
    if (eos) break;
    if (word == -1) continue;
    ++*word_count;
    // The subsampling randomly discards frequent words while keeping the ranking same
    if (! language_model.subsample(word)) {
      continue;
    }
    sen[sentence_length] = word;
    ++sentence_length;
    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
  }
  return sentence_length;
}

void TrainModelThread(WordContextFactorization& factorization,
                      DiscreteSamplingStrategy<NaiveLanguageModel>& neg_sampling_strategy,
                      const NaiveLanguageModel& language_model,
                      SGD& sgd,
                      const DynamicContextStrategy& ctx_strategy) {
  long long
    sentence_length = 0,
    input_word_position = 0,
    word_count = 0,
    last_word_count = 0,
    local_iter = iter;
  long long sen[MAX_SENTENCE_LENGTH + 1];
  real *input_word_gradient = (real *)calloc(factorization.get_embedding_dim(), sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      update_progress(language_model, sgd);
      last_word_count = word_count;
    }
    char eof = 0;
    if (input_word_position >= sentence_length) {
      sentence_length = read_new_sentence(fi, language_model, &word_count,
                                          sen, &eof);
      input_word_position = 0;
    }
    if (eof || (word_count > language_model.total() / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, 0, SEEK_SET);
      continue;
    }
    long long input_word = sen[input_word_position];
    if (input_word == -1) continue;
    auto ctx = ctx_strategy.size(
      input_word_position,
      (sentence_length - 1) - input_word_position);
    const real alpha = sgd.get_rho(input_word);

    for (long long output_word_position = input_word_position - ctx.first;
        output_word_position <= input_word_position + ctx.second;
        ++output_word_position) if (output_word_position != input_word_position) {
      long long output_word = sen[output_word_position];
      if (output_word == -1) continue;
      zero_vector(factorization.get_embedding_dim(), input_word_gradient);
      // NEGATIVE SAMPLING
      if (negative > 0) for (long long d = 0; d < negative + 1; d++) {
        long long target_word, is_output;
        if (d == 0) {
          target_word = output_word;
          is_output = 1;
        } else {
          target_word = neg_sampling_strategy.sample_idx(language_model);
          if (target_word == output_word) continue;
          is_output = 0;
        }
        real f = cblas_sdot(factorization.get_embedding_dim(),
                      factorization.get_word_embedding(input_word), 1,
                      factorization.get_context_embedding(target_word), 1);
        real gradient_scale = (is_output - fast_sigmoid(f)) * alpha;
        cblas_saxpy(factorization.get_embedding_dim(),
              gradient_scale,
              factorization.get_context_embedding(target_word), 1,
              input_word_gradient, 1);
        cblas_saxpy(factorization.get_embedding_dim(),
              gradient_scale,
              factorization.get_word_embedding(input_word), 1,
              factorization.get_context_embedding(target_word), 1);
      }
      // Learn weights input -> hidden
      cblas_saxpy(factorization.get_embedding_dim(),
            1,
            input_word_gradient, 1,
            factorization.get_word_embedding(input_word), 1);
    }
    input_word_position++;
    sgd.step(input_word);
  }
  fclose(fi);
  free(input_word_gradient);
}

void TrainModel(NaiveLanguageModel& language_model, real alpha, long long embedding_size, int window) {
  printf("Starting training using file %s\n", train_file);
  if (read_vocab_file[0] != 0) ReadVocab(language_model); else LearnVocabFromTrainFile(language_model);
  if (save_vocab_file[0] != 0) SaveVocab(language_model);
  if (output_file[0] == 0) return;

  size_t lm_size(language_model.size());
  size_t lm_total(language_model.total());
  ExponentCountNormalizer normalizer(0.75);
  SGD sgd(lm_size, lm_total * iter, alpha, 0.0001 * alpha);
  WordContextFactorization factorization(lm_size, embedding_size);
  DiscreteSamplingStrategy<NaiveLanguageModel> neg_sampling_strategy(Discretization(normalizer.normalize(language_model.counts()), table_size));
  DynamicContextStrategy ctx_strategy(window);
  start = clock();
  TrainModelThread(factorization,
                   neg_sampling_strategy,
                   language_model,
                   sgd,
                   ctx_strategy);
  FILE *fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%zu %lld\n", lm_size, embedding_size);
    for (long a = 0; a < lm_size; a++) {
      fprintf(fo, "%s ", language_model.reverse_lookup(a).c_str());
      if (binary) for (long b = 0; b < embedding_size; b++) fwrite(factorization.get_word_embedding(a) + b, sizeof(real), 1, fo);
      else for (long b = 0; b < embedding_size; b++) fprintf(fo, "%lf ", factorization.get_word_embedding(a)[b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(lm_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * embedding_size, sizeof(real));
    for (long a = 0; a < lm_size; a++) cl[a] = a % clcn;
    for (long a = 0; a < iter; a++) {
      for (long b = 0; b < clcn * embedding_size; b++) cent[b] = 0;
      for (long b = 0; b < clcn; b++) centcn[b] = 1;
      for (long c = 0; c < lm_size; c++) {
        for (long d = 0; d < embedding_size; d++) cent[embedding_size * cl[c] + d] += factorization.get_word_embedding(c)[d];
        centcn[cl[c]]++;
      }
      for (long b = 0; b < clcn; b++) {
        closev = 0;
        for (long c = 0; c < embedding_size; c++) {
          cent[embedding_size * b + c] /= centcn[b];
          closev += cent[embedding_size * b + c] * cent[embedding_size * b + c];
        }
        closev = sqrt(closev);
        for (long c = 0; c < embedding_size; c++) cent[embedding_size * b + c] /= closev;
      }
      for (long c = 0; c < lm_size; c++) {
        closev = -10;
        closeid = 0;
        for (long d = 0; d < clcn; d++) {
          x = 0;
          for (long b = 0; b < embedding_size; b++) x += cent[embedding_size * d + b] * factorization.get_word_embedding(c)[b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (long a = 0; a < lm_size; a++) fprintf(fo, "%s %d\n", language_model.reverse_lookup(a).c_str(), cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -binary 0 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  int i;
  int window = 5;
  real alpha = 0.025;
  real sample = 1e-3;
  long long embedding_size = 100;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) embedding_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if (num_threads != 1) {
    printf("Must have num_threads = 1\n");
    exit(1);
  }
  NaiveLanguageModel language_model(sample);
  TrainModel(language_model, alpha, embedding_size, window);
  return 0;
}
