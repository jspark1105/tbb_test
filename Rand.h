#pragma once

#include "AlignedVec.h"

// Fill with uniform random numbers
// duplicate is meaningful only for integral types
extern void randFill(aligned_vector<float> &vec,
                     const float low, const float high);
extern void randFill(aligned_vector<double> &vec,
                     const double low, const double high);
extern void randFill(aligned_vector<int> &vec,
                     const int low, const int high,
                     bool duplicates = true);
extern void randFill(aligned_vector<int64_t> &vec,
                     const int low, const int high,
                     bool duplicates = true);
extern void randFill(float *array, size_t len,
                     const float low, const float high,
                     bool duplicates = true);
extern void randFill(double *array, size_t len,
                     const double low, const double high,
                     bool duplicates = true);
extern void randFill(int *array, int len,
                     const int low, const int high,
                     bool duplicates = true);
extern void randFill(int64_t *array, int len,
                     const int low, const int high,
                     bool duplicates = true);
