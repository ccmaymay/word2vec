// Modified from athena/_math.cpp in athena:
// https://github.com/cjmay/athena/blob/master/athena/_math.cpp
//
// Copyright 2012-2017 Johns Hopkins University Human Language Technology
// Center of Excellence (JHU HLTCOE). All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.
// 
// The views and conclusions contained in the software and documentation
// are those of the authors and should not be interpreted as representing
// official policies, either expressed or implied, of the copyright
// holders.

#include "_math.h"
#include <cmath>
#include <cstdlib>
#include <new>
#include <climits>
#include <vector>
#include <utility>
#include <random>
#include <cstring>

#define omp_get_num_threads() 1
#define omp_get_thread_num() 0


using namespace std;


static vector<PRNG> prngs;


void seed(unsigned int s) {
  prngs.clear();
  int num_threads = 1;
  #pragma omp parallel default(shared)
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
  }
  for (int t = 0; t < num_threads; ++t) {
    prngs.push_back(PRNG(s + t));
  }
}

void seed_default() {
  seed(random_device()());
}

PRNG& get_urng() {
  if (prngs.empty()) {
    seed(0);
  }
  return prngs[omp_get_thread_num()];
}


//
// CountNormalizer
//


CountNormalizer::CountNormalizer(float exponent, float offset):
    _exponent(exponent),
    _offset(offset) { }

vector<float> CountNormalizer::normalize(const vector<size_t>& counts) const {
  vector<float> probabilities(counts.size(), 0);
  float normalizer = 0;
  for (size_t i = 0; i < probabilities.size(); ++i) {
    probabilities[i] = pow(counts[i] + _offset, _exponent);
    normalizer += probabilities[i];
  }
  for (size_t i = 0; i < probabilities.size(); ++i) {
    probabilities[i] /= normalizer;
  }
  return probabilities;
}
