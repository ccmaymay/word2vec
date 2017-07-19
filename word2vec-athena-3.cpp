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

#include <athena/athena/_math.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

using namespace std;

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
long long vocab_max_size = 1000, vocab_size = 0, embedding_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
clock_t start;

int negative = 5;
const int table_size = 1e8;

void saxpy(int n, real alpha, const real* x, real* y) {
  for (int i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

real sdot(int n, const real* x, const real* y) {
  real sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}

real fast_exp(const real* expTable, real f) {
  if (f > MAX_EXP) return 1;
  else if (f < -MAX_EXP) return 0;
  else return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
}

long long draw_negative_sample(const ReservoirSampler<long>& table) {
  long long neg_sample = table.sample();
  if (neg_sample == 0) {
    uniform_int_distribution<unsigned long long> d(1, vocab_size - 1);
    return d(get_urng());
  } else {
    return neg_sample;
  }
}

void zero_vector(long long n, real* v) {
  memset(v, 0, n * sizeof(real));
}

void InitUnigramTable(struct vocab_word* vocab, ReservoirSampler<long>& table) {
  double train_words_pow = 0;
  double power = 0.75;
  for (int a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  for (int a = 0; a < vocab_size; a++) {
    double probability = pow(vocab[a].cn, power) / train_words_pow;
    for (int i = 0; i <= probability * table_size; ++i) {
      table.insert(a);
    }
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
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
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(const char *word) {
  unsigned long long hash = 0;
  for (unsigned long long a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(const struct vocab_word* vocab, const int* vocab_hash, const char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(const struct vocab_word* vocab, const int* vocab_hash, FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  return SearchVocab(vocab, vocab_hash, word);
}

// Adds a word to the vocabulary
int AddWordToVocab(struct vocab_word** vocab, int* vocab_hash, const char *word) {
  unsigned int length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  (*vocab)[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy((*vocab)[vocab_size].word, word);
  (*vocab)[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    *vocab = (struct vocab_word *)realloc((*vocab), vocab_max_size * sizeof(struct vocab_word));
  }
  unsigned int hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(struct vocab_word** vocab, int* vocab_hash) {
  // Sort the vocabulary and keep </s> at the first position
  qsort(*vocab + 1, vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (int a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  int size = vocab_size;
  train_words = 0;
  for (int a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (((*vocab)[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free((*vocab)[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      unsigned int hash=GetWordHash((*vocab)[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += (*vocab)[a].cn;
    }
  }
  *vocab = (struct vocab_word *)realloc((*vocab), (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (int a = 0; a < vocab_size; a++) {
    (*vocab)[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    (*vocab)[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct vocab_word* vocab, int* vocab_hash) {
  int b = 0;
  for (int a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (int a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (int a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    unsigned int hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree(struct vocab_word* vocab) {
  long long b;
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (long long a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (long long a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  long long pos1 = vocab_size - 1;
  long long pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (long long a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    long long min1i;
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    long long min2i;
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (long long a = 0; a < vocab_size; a++) {
    b = a;
    char code[MAX_CODE_LENGTH];
    long long point[MAX_CODE_LENGTH];
    long long i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile(struct vocab_word** vocab, int* vocab_hash) {
  long long wc = 0;
  for (long long a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  FILE *fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab(vocab, vocab_hash, (char *)"</s>");
  while (1) {
    char word[MAX_STRING];
    char eof = 0;
    ReadWord(word, fin, &eof);
    if (eof) break;
    train_words++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    long long i = SearchVocab(*vocab, vocab_hash, word);
    if (i == -1) {
      long long a = AddWordToVocab(vocab, vocab_hash, word);
      (*vocab)[a].cn = 1;
    } else (*vocab)[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab(*vocab, vocab_hash);
  }
  SortVocab(vocab, vocab_hash);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab(struct vocab_word* vocab) {
  FILE *fo = fopen(save_vocab_file, "wb");
  for (long long i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab(struct vocab_word** vocab, int* vocab_hash) {
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (long long a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  long long i = 0;
  while (1) {
    char word[MAX_STRING];
    char eof = 0;
    ReadWord(word, fin, &eof);
    if (eof) break;
    long long a = AddWordToVocab(vocab, vocab_hash, word);
    char c;
    fscanf(fin, "%lld%c", &(*vocab)[a].cn, &c);
    i++;
  }
  SortVocab(vocab, vocab_hash);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
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

void InitNet(real** input_embeddings, real** output_embeddings, struct vocab_word* vocab) {
  posix_memalign((void **)input_embeddings, 128, (long long)vocab_size * embedding_size * sizeof(real));
  if (*input_embeddings == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (negative>0) {
    posix_memalign((void **)output_embeddings, 128, (long long)vocab_size * embedding_size * sizeof(real));
    if (*output_embeddings == NULL) {printf("Memory allocation failed\n"); exit(1);}
    zero_vector(vocab_size * embedding_size, *output_embeddings);
  }
  uniform_real_distribution<float> d(-0.5, 0.5);
  for (long long a = 0; a < vocab_size; a++) for (long long b = 0; b < embedding_size; b++) {
    (*input_embeddings)[a * embedding_size + b] = d(get_urng()) / embedding_size;
  }
  CreateBinaryTree(vocab);
}

void update_progress_and_learning_rate() {
  if ((debug_mode > 1)) {
    clock_t now = clock();
    printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
     word_count_actual / (real)(iter * train_words + 1) * 100,
     word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
    fflush(stdout);
  }
  alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
  if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
}

long long read_new_sentence(FILE* fi, const struct vocab_word* vocab,
                            const int* vocab_hash, long long* word_count,
                            long long* sen, char* eof) {
  long long sentence_length = 0;
  uniform_real_distribution<float> d(0, 1);
  while (1) {
    long long word = ReadWordIndex(vocab, vocab_hash, fi, eof);
    if (*eof) break;
    if (word == -1) continue;
    ++*word_count;
    if (word == 0) break;
    // The subsampling randomly discards frequent words while keeping the ranking same
    if (sample > 0) {
      real threshold =
        (sqrt(vocab[word].cn / (sample * train_words)) + 1) *
        (sample * train_words) / vocab[word].cn;
      if (d(get_urng()) > threshold)
        continue;
    }
    sen[sentence_length] = word;
    ++sentence_length;
    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
  }
  return sentence_length;
}

void TrainModelThread(real* input_embeddings,
                      real* output_embeddings,
                      const ReservoirSampler<long>& table,
                      real* expTable,
                      struct vocab_word** vocab,
                      int* vocab_hash) {
  long long
    sentence_length = 0,
    output_word_position = 0,
    word_count = 0,
    last_word_count = 0,
    local_iter = iter;
  long long sen[MAX_SENTENCE_LENGTH + 1];
  real *input_word_gradient = (real *)calloc(embedding_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      update_progress_and_learning_rate();
      last_word_count = word_count;
    }
    char eof = 0;
    if (output_word_position >= sentence_length) {
      sentence_length = read_new_sentence(fi, *vocab, vocab_hash, &word_count,
                                          sen, &eof);
      output_word_position = 0;
    }
    if (eof || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, 0, SEEK_SET);
      continue;
    }
    long long output_word = sen[output_word_position];
    if (output_word == -1) continue;
    uniform_int_distribution<long long> dyn_window_offset_d(0, window - 1);
    long long dyn_window_offset = dyn_window_offset_d(get_urng());

    for (long long a = dyn_window_offset; a < window * 2 + 1 - dyn_window_offset; a++) if (a != window) {
      long long input_word_position = output_word_position - window + a;
      if (input_word_position < 0) continue;
      if (input_word_position >= sentence_length) continue;
      long long input_word = sen[input_word_position];
      if (input_word == -1) continue;
      long long input_embeddings_offset = input_word * embedding_size;
      zero_vector(embedding_size, input_word_gradient);
      // NEGATIVE SAMPLING
      if (negative > 0) for (long long d = 0; d < negative + 1; d++) {
        long long target_word, is_output;
        if (d == 0) {
          target_word = output_word;
          is_output = 1;
        } else {
          target_word = draw_negative_sample(table);
          if (target_word == output_word) continue;
          is_output = 0;
        }
        long long output_embeddings_offset = target_word * embedding_size;
        real f = sdot(embedding_size,
                      input_embeddings + input_embeddings_offset,
                      output_embeddings + output_embeddings_offset);
        real gradient_scale = (is_output - fast_exp(expTable, f)) * alpha;
        saxpy(embedding_size,
              gradient_scale,
              output_embeddings + output_embeddings_offset,
              input_word_gradient);
        saxpy(embedding_size,
              gradient_scale,
              input_embeddings + input_embeddings_offset,
              output_embeddings + output_embeddings_offset);
      }
      // Learn weights input -> hidden
      saxpy(embedding_size,
            1,
            input_word_gradient,
            input_embeddings + input_embeddings_offset);
    }
    output_word_position++;
  }
  fclose(fi);
  free(input_word_gradient);
}

void TrainModel(real* expTable, struct vocab_word** vocab, int* vocab_hash) {
  real *input_embeddings, *output_embeddings;
  ReservoirSampler<long> table(table_size);
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(vocab, vocab_hash); else LearnVocabFromTrainFile(vocab, vocab_hash);
  if (save_vocab_file[0] != 0) SaveVocab(*vocab);
  if (output_file[0] == 0) return;
  InitNet(&input_embeddings, &output_embeddings, *vocab);
  InitUnigramTable(*vocab, table);
  start = clock();
  TrainModelThread(input_embeddings,
                   output_embeddings,
                   table,
                   expTable,
                   vocab,
                   vocab_hash);
  FILE *fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, embedding_size);
    for (long a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", (*vocab)[a].word);
      if (binary) for (long b = 0; b < embedding_size; b++) fwrite(&input_embeddings[a * embedding_size + b], sizeof(real), 1, fo);
      else for (long b = 0; b < embedding_size; b++) fprintf(fo, "%lf ", input_embeddings[a * embedding_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * embedding_size, sizeof(real));
    for (long a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (long a = 0; a < iter; a++) {
      for (long b = 0; b < clcn * embedding_size; b++) cent[b] = 0;
      for (long b = 0; b < clcn; b++) centcn[b] = 1;
      for (long c = 0; c < vocab_size; c++) {
        for (long d = 0; d < embedding_size; d++) cent[embedding_size * cl[c] + d] += input_embeddings[c * embedding_size + d];
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
      for (long c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (long d = 0; d < clcn; d++) {
          x = 0;
          for (long b = 0; b < embedding_size; b++) x += cent[embedding_size * d + b] * input_embeddings[c * embedding_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (long a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", (*vocab)[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  free(input_embeddings);
  if (negative > 0) {
    free(output_embeddings);
  }
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
  struct vocab_word *vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  int *vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  real* expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (int j = 0; j < EXP_TABLE_SIZE; j++) {
    expTable[j] = exp((j / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[j] = expTable[j] / (expTable[j] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel(expTable, &vocab, vocab_hash);
  free(expTable);
  free(vocab_hash);
  for (long long a = 0; a < vocab_size; a++) {
    free(vocab[a].word);
    free(vocab[a].code);
    free(vocab[a].point);
  }
  free(vocab);
  return 0;
}
