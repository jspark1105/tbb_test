#include "Rand.h"

#include <unordered_map>
#include <unordered_set>
#include <random>

#include <mkl.h>

std::default_random_engine eng;

#define USE_MKL_RNG

#ifdef USE_MKL_RNG
VSLStreamStatePtr getRngStream()
{
  static VSLStreamStatePtr rng_stream;
  static bool initialized = false;

  if (!initialized) {
    vslNewStream(&rng_stream, VSL_BRNG_NIEDERR, 3);
    initialized = true;
  }

  return rng_stream;
}
#endif

void randFill(float *vec, size_t len, const float low, const float high,
              bool duplicates) {

#ifdef USE_MKL_RNG
  if (len > (1 << 29)) {
    for (size_t i = 0; i < len; i += (1 << 29)) {
      int l = std::min(1UL << 29, len - i);
      vsRngUniform(
        VSL_RNG_METHOD_UNIFORM_STD, getRngStream(), l, vec + i, low, high);
    }
  }
  else {
    vsRngUniform(
      VSL_RNG_METHOD_UNIFORM_STD, getRngStream(), len, vec, low, high);
  }
#else
  std::uniform_real_distribution<float> dis(low, high);
  for (size_t i = 0; i < len; ++i) {
    vec[i] = dis(eng);
  }
#endif
}

void randFill(double *vec, size_t len, const double low, const double high,
              bool duplicates) {
#ifdef USE_MKL_RNG
  if (len > (1 << 29)) {
    for (size_t i = 0; i < len; i += (1 << 29)) {
      int l = std::min(1UL << 29, len - i);
      vdRngUniform(
        VSL_RNG_METHOD_UNIFORM_STD, getRngStream(), l, vec + i, low, high);
    }
  }
  else {
    vdRngUniform(
      VSL_RNG_METHOD_UNIFORM_STD, getRngStream(), len, vec, low, high);
  }
#else
  std::uniform_real_distribution<double> dis(low, high);
  for (size_t i = 0; i < len; ++i) {
    vec[i] = dis(eng);
  }
#endif
}

void randFill(
    aligned_vector<float> &vec, const float low, const float high) {
  randFill(vec.data(), vec.size(), low, high);
}

void randFill(
    aligned_vector<double> &vec, const double low, const double high) {
  randFill(vec.data(), vec.size(), low, high);
}

template<typename T>
void randFill_(T *array, int len, const int low, const int high,
              bool duplicates) {
  if (duplicates == true) {
    std::uniform_int_distribution<T> dis(low, high);
    for (int i = 0; i < len; ++i) {
      array[i] = dis(eng);
    }
  } else {
    // no duplicates
    // produce a list
    assert(high - low >= len); // can't sample without duplicates

    // An array with logical length high - low + 1
    // If identity_map[k] doesn't exist, it implies identity_map[k] = k
    std::unordered_map<int, T> identity_map;

    for (auto i = 0; i < len; ++i) {
      int j = (eng() % (high - low + 1 - i)) + i;

      // swap identity_map[i] and identity_map[j], and put
      // identity_map[i] to array[i].
      array[i] =
        (identity_map.find(j) == identity_map.end() ? j : identity_map[j]) +
        low;
      identity_map[j] =
        identity_map.find(i) == identity_map.end() ? i : identity_map[i];
    }

    std::unordered_set<T> selected;
    for (int i = 0; i < len; ++i) {
      assert(array[i] >= low);
      assert(array[i] <= high);
      assert(selected.find(array[i]) == selected.end());
      selected.insert(array[i]);
    }
  }
}

void randFill(int *array, int len, const int low, const int high,
              bool duplicates) {
  return randFill_(array, len, low, high, duplicates);
}

void randFill(int64_t *array, int len, const int low, const int high,
              bool duplicates) {
  return randFill_(array, len, low, high, duplicates);
}

void randFill(aligned_vector<int> &vec, const int low, const int high,
              bool duplicates) {
  return randFill(vec.data(), vec.size(), low, high, duplicates);
}

void randFill(aligned_vector<int64_t> &vec, const int low,
              const int high, bool duplicates) {
  return randFill(vec.data(), vec.size(), low, high, duplicates);
}
