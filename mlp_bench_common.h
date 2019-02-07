#pragma once

#include <memory>
#include <vector>

#include "Matrix.h"
#include "Partition.h"

constexpr int PAD = 16;
constexpr int CACHE_LINE_LEN = 16;

enum Breakdown {
  FWD = 0,
  WGT_GRAD,
  BWD,
  WGT_UPDATE,
  WGT_UPDATE_REDUCE_SCATTER,
  WGT_UPDATE_ALLGATHER,
  NUM_BREAKDOWNS,
};

// Every other nfeatures should be divisible by 16 to make all-reduce work
// for now.
// To eliminate this constraint, we need pading after each socket's copy of
// weight.
constexpr int nfeatures[] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1};
// int nfeatures[] = { 512, 512 };
constexpr int nlayers = sizeof(nfeatures) / sizeof(nfeatures[0]) - 1;
constexpr int MAX_NUM_THREADS = 1024;
extern int nthreads_per_socket, nthreads;

// Being careful not to have false sharing
constexpr int NUM_BREAKDOWNS_ROUNDED_UP =
    (NUM_BREAKDOWNS + CACHE_LINE_LEN - 1) / CACHE_LINE_LEN * CACHE_LINE_LEN;
extern double sum_times[MAX_NUM_THREADS][nlayers][NUM_BREAKDOWNS_ROUNDED_UP],
    sum_flops[nlayers][NUM_BREAKDOWNS];
extern double wall_clock_time;
constexpr int NWARMUP = 16, NITER = 256;

constexpr int nthreads_per_socket_for_allreduce[29] = {
    0, 0, 0, 0, 0,
    1, // total 5, 1 for all-reduce 4 for gemm
    1, 1, 1, 1,
    2, // total 10, 2 for all-reduce 8 for gemm
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, // total 20, 4 for all-reduce 16 for gemm
    4, 4, 4, 4, 4, 4, 4, 4,
};

// 2 rings going to the apposite directions.
extern std::vector<std::array<int, 8>> rings;

extern std::unique_ptr<Matrix<float, PAD>> weights[nlayers],
    weight_grads[nlayers], weight_grad_push_bufs[nlayers],
    activations[nlayers + 2];

void init_matrices();

void report_timing();

void print_checksum();

void get_my_ring_info(
    int sid,
    int ask,
    int* idx_in_ring,
    int* prev_sid,
    int* next_sid);
