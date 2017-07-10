// Modified from athena/_math.h in athena:
// https://github.com/cjmay/athena/blob/master/athena/_math.h
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

#ifndef ATHENA__MATH_H
#define ATHENA__MATH_H


#include <cstddef>
#include <cmath>
#include <vector>
#include <utility>
#include <random>
#include <iostream>
#include <unordered_set>
#include <memory>


typedef std::linear_congruential_engine<size_t,25214903917ull,11ull,1ull<<48>
        PRNG;


// Seed the random number generator(s).
void seed(unsigned int s);


// Seed the random number generator(s) randomly.
void seed_default();


// Get thread's random number generator.
PRNG& get_urng();


class CountNormalizer {
  float _exponent;
  float _offset;

  public:
    CountNormalizer(float exponent = 1, float offset = 0);
    virtual std::vector<float> normalize(const std::vector<size_t>&
                                            counts) const;
    virtual ~CountNormalizer() { }

  private:
    CountNormalizer(const CountNormalizer& count_normalizer);
};


template <typename T>
class AliasSampler;

template <typename T>
class AliasSampler {
  T _size;
  std::vector<T> _alias_table;
  std::vector<float> _probability_table;

  public:
    AliasSampler(const std::vector<float>& probabilities);
    virtual T sample() const;
    virtual ~AliasSampler() { }

    AliasSampler(T size,
                 std::vector<T>&& alias_table,
                 std::vector<float>&& probability_table):
      _size(size),
      _alias_table(std::forward<std::vector<T> >(alias_table)),
      _probability_table(
        std::forward<std::vector<float> >(probability_table)) { }

  private:
    AliasSampler(const AliasSampler& alias_sampler);
};


template <typename T>
class ReservoirSampler;

template <typename T>
class ReservoirSampler {
  size_t _size, _filled_size, _count;
  std::vector<T> _reservoir;

  public:
    ReservoirSampler(size_t size);
    virtual T sample() const {
      std::uniform_int_distribution<size_t> d(0, _filled_size - 1);
      return _reservoir[d(get_urng())];
    }
    virtual const T& operator[](size_t idx) const {
      return _reservoir[idx];
    }
    virtual size_t size() const { return _size; }
    virtual size_t filled_size() const { return _filled_size; }
    virtual T insert(T val);
    virtual void clear();
    virtual ~ReservoirSampler() { }

    ReservoirSampler(size_t size, size_t filled_size, size_t count,
                     std::vector<T>&& reservoir):
      _size(size),
      _filled_size(filled_size),
      _count(count),
      _reservoir(std::forward<std::vector<T> >(reservoir)) { }

    ReservoirSampler(ReservoirSampler<T>&& reservoir_sampler);

  private:
    ReservoirSampler(const ReservoirSampler<T>& reservoir_sampler);
};


template <typename T>
class Discretization;

template <typename T>
class Discretization {
  std::vector<T> _samples;

  public:
    Discretization(const std::vector<float>& probabilities,
                   size_t num_samples);
    virtual T sample() const {
      std::uniform_int_distribution<size_t> d(0, _samples.size() - 1);
      return _samples[d(get_urng())];
    }
    virtual const T& operator[](size_t idx) const { return _samples[idx]; }
    virtual size_t num_samples() const { return _samples.size(); }
    virtual ~Discretization() { }

    Discretization(std::vector<T>&& samples):
      _samples(std::forward<std::vector<T> >(samples)) { }

  private:
    Discretization(const Discretization& discretization);
};


//
// AliasSampler
//


template <typename T>
AliasSampler<T>::AliasSampler(const std::vector<float>& probabilities):
    _size(probabilities.size()),
    _alias_table(probabilities.size(), 0),
    _probability_table(probabilities.size(), 0.) {
  std::unordered_set<T> underfull, overfull;
  for (T i = 0; i < _size; ++i) {
    const float mass = _size * probabilities[i];
    _alias_table[i] = i;
    _probability_table[i] = mass;
    if (mass < 1.) {
      underfull.insert(i);
    } else if (mass > 1.) {
      overfull.insert(i);
    }
  }

  while (! (overfull.empty() || underfull.empty())) {
    auto underfull_it = underfull.begin();
    const T underfull_idx = *underfull_it;

    auto overfull_it = overfull.begin();
    const T overfull_idx = *overfull_it;

    _alias_table[underfull_idx] = overfull_idx;
    _probability_table[overfull_idx] -= 1. - _probability_table[underfull_idx];

    // set underfull_idx to exactly full
    underfull.erase(underfull_it);

    // set overfull_idx to underfull or exactly full, if appropriate
    if (_probability_table[overfull_idx] < 1.) {
      overfull.erase(overfull_it);
      underfull.insert(overfull_idx);
    } else if (_probability_table[overfull_idx] == 1.) {
      overfull.erase(overfull_it);
    }
  }

  // overfull and underfull may contain masses negligibly close to 1
  // due to floating-point error
  for (auto it = overfull.cbegin(); it != overfull.cend(); ++it) {
    _alias_table[*it] = *it;
    _probability_table[*it] = 1.;
  }
  for (auto it = underfull.cbegin(); it != underfull.cend(); ++it) {
    _alias_table[*it] = *it;
    _probability_table[*it] = 1.;
  }
}

template <typename T>
T AliasSampler<T>::sample() const {
  std::uniform_int_distribution<T> d(0, _size - 1);
  const T i = d(get_urng());
  if (_probability_table[i] == 1) {
    return i;
  } else {
    std::bernoulli_distribution alias_d(_probability_table[i]);
    return (alias_d(get_urng()) ? i : _alias_table[i]);
  }
}


//
// ReservoirSampler
//


template <typename T>
ReservoirSampler<T>::ReservoirSampler(size_t size):
    _size(size),
    _filled_size(0),
    _count(0),
    _reservoir(size) { }

template <typename T>
T ReservoirSampler<T>::insert(T val) {
  if (_filled_size < _size) {
    // reservoir not yet at capacity, insert val
    _reservoir[_filled_size] = val;
    ++_filled_size;
    ++_count;
    return val;
  } else {
    // reservoir at capacity, insert val w.p. _size/(_count+1)
    // (+1 is for val)
    std::uniform_int_distribution<size_t> d(0, _count);
    const size_t idx = d(get_urng());
    if (idx < _size) {
      const T prev_val = _reservoir[idx];
      _reservoir[idx] = val;
      ++_count;
      return prev_val;
    } else {
      ++_count;
      return val;
    }
  }
}

template <typename T>
void ReservoirSampler<T>::clear() {
  _filled_size = 0;
  _count = 0;
}

template <typename T>
ReservoirSampler<T>::ReservoirSampler(ReservoirSampler<T>&& reservoir_sampler):
    ReservoirSampler(reservoir_sampler._size,
                     reservoir_sampler._filled_size,
                     reservoir_sampler._count,
                     std::forward<std::vector<T> >(
                       reservoir_sampler._reservoir)) {
  reservoir_sampler._size = 0;
  reservoir_sampler._filled_size = 0;
  reservoir_sampler._count = 0;
  reservoir_sampler._reservoir.clear();
}


//
// Discretization
//


template <typename T>
Discretization<T>::Discretization(const std::vector<float>& probabilities,
                                  size_t num_samples):
    _samples(num_samples, -1) {
  if (! probabilities.empty()) {
    size_t i = 0, j = 0;
    float cum_mass = probabilities[j];

    while (i < num_samples) {
      // add sample before checking bounds: favor weights near beginning
      // of input
      _samples[i] = j;
      ++i;

      if (i / (float) num_samples > cum_mass) {
        ++j;
        if (j == probabilities.size())
          break;

        cum_mass += probabilities[j];
      }
    }

    for (; i < num_samples; ++i)
      _samples[i] = probabilities.size() - 1;
  }
}


#endif
