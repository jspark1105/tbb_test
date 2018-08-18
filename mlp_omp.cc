#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <mkl.h>
#include <numa.h>
#include <omp.h>
#include <x86intrin.h>

#include "Matrix.h"
#include "Partition.h"
#include "Rand.h"

//#define PRINT_PER_LAYER_PERFORMANCE

using namespace std;

constexpr int PAD = 16;
constexpr int CACHE_LINE_LEN = 16;

Matrix<float, PAD> *
  create_matrix_with_numa_aware_allocation(int nrows, int ncols)
{
  Matrix<float, PAD> *matrix = new Matrix<float, PAD>(nrows, ncols);
#pragma omp parallel
  {
    int sid = get_socket_num();
    auto bm = numa_allocate_nodemask();
    numa_bitmask_clearall(bm);
    numa_bitmask_setbit(bm, sid);
    numa_bind(bm);
    numa_bitmask_free(bm);

    pair<int, int> i_range = get_partition(nrows);
    for (int i = i_range.first; i < i_range.second; ++i) {
      for (int j = 0; j < ncols; ++j) {
        (*matrix)(i, j) = (i + j) % 31 - 15;
      }
    }
  }
  return matrix;
}

enum Breakdown {
  FWD = 0,
  WGT_GRAD,
  BWD,
  NUM_BREAKDOWNS,
};

// Every other nfeatures should be divisible by 16 to make all-reduce work
// for now.
// To eliminate this constraint, we need pading after each socket's copy of
// weight.
int nfeatures[] = { 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1 };
//int nfeatures[] = { 512, 512 };
constexpr int nlayers = sizeof(nfeatures)/sizeof(nfeatures[0]) - 1;
constexpr int MAX_NUM_THREADS = 1024;

// Being careful not to have false sharing
constexpr int NUM_BREAKDOWNS_ROUNDED_UP =
  (NUM_BREAKDOWNS + CACHE_LINE_LEN - 1) / CACHE_LINE_LEN * CACHE_LINE_LEN;
double
  sum_times[MAX_NUM_THREADS][nlayers][NUM_BREAKDOWNS_ROUNDED_UP] = { 0 },
  sum_flops[nlayers][NUM_BREAKDOWNS] = { 0 };

int main(int argc, char **argv)
{
  if (argc != 3) {
    fprintf(stderr, "%s <nsockets> <nthreads_per_socket>\n", argv[0]);
    exit(1);
  }

  nsockets = atoi(argv[1]);
  int nthreads_per_socket = atoi(argv[2]);
  int nthreads = nsockets * nthreads_per_socket;
  omp_set_num_threads(nthreads);

  int batch_size = 1024 * nsockets; // weak-scaling with nsockets

  unique_ptr<Matrix<float, PAD>>
    weights[nlayers], weight_grads[nlayers],
    activations[nlayers + 2];

  /////////////////////////////////////////////////////////////////////////////
  // allocate memory and "first-touch" for NUMA-aware allocation
  for (int l = 0; l < nlayers; ++l) {
    activations[l].reset(
      create_matrix_with_numa_aware_allocation(batch_size, nfeatures[l]));

    // Weights and their gradients are replicated at each socket.
    weights[l].reset(
      create_matrix_with_numa_aware_allocation(
        nsockets * nfeatures[l + 1], nfeatures[l]));

    weight_grads[l].reset(
      create_matrix_with_numa_aware_allocation(
        nsockets * nfeatures[l + 1], nfeatures[l]));
  }
  activations[nlayers].reset(
    create_matrix_with_numa_aware_allocation(batch_size, nfeatures[nlayers]));
  // this extra is to avoid overwriting the input
  activations[nlayers + 1].reset(
    create_matrix_with_numa_aware_allocation(batch_size, nfeatures[0]));

  /////////////////////////////////////////////////////////////////////////////
  // initialize values (only done by the master thread to be deterministic)

  // initialize input
  activations[0]->randFill(0.f, 1.f);

  // initialize weights
  for (int l = 0; l < nlayers; ++l) {
    for (int i = 0; i < nfeatures[l + 1]; ++i) {
      randFill(weights[l]->rawData(i, 0), weights[l]->ncols(), -0.1f, 0.1f);
    }

    for (int s = 1; s < nsockets; ++s) {
      memcpy(
        weights[l]->rawData(s * nfeatures[l + 1], 0),
        weights[l]->rawData(),
        nfeatures[l + 1] * weights[l]->ld() * sizeof(float));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Main computation

  constexpr int NWARMUP = 2, NITER = 256;
#pragma omp parallel
  {
    int sid = get_socket_num();
    int tid = omp_get_thread_num();

    auto bm = numa_allocate_nodemask();
    numa_bitmask_clearall(bm);
    numa_bitmask_setbit(bm, sid);
    numa_bind(bm);
    numa_bitmask_free(bm);

    for (int it = 0; it < NWARMUP + NITER; ++it) {
      // forward
      for (int l = 0; l < nlayers; ++l) {

        double t0 = dsecnd();

        int m = batch_size, n = nfeatures[l + 1], k = nfeatures[l];
        int m_begin, m_end, n_begin, n_end;
        // forward gemm performs well with aspect ratio
        // (m_end - m_begin) ~= 32 * (n_end - n_begin)
        float aspect_ratio = 4.;
        /*if (n == 512 && k == 512) {
          aspect_ratio = 2.;
        }*/
        get_2dpartition(
            &m_begin,
            &m_end,
            &n_begin,
            &n_end,
            m,
            n,
            aspect_ratio,
            false /* m_align */);
        if (0 == it && 0 == tid) {
          int mb = (m / nsockets + m_end - m_begin - 1) / (m_end - m_begin);
          int nb = (n + n_end - n_begin - 1) / (n_end - n_begin);
          printf(
              "fwd m %d n %d k %d bm %d bn %d bk %d mb %d nb %d kb %d\n",
              m,
              n,
              k,
              m_end - m_begin,
              n_end - n_begin,
              k,
              mb,
              nb,
              1);
        }

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            m_end - m_begin,
            n_end - n_begin,
            k,
            1.0f,
            activations[l]->rawData(m_begin, 0),
            activations[l]->ld(),
            weights[l]->rawData(sid * n + n_begin, 0),
            weights[l]->ld(),
            0.0f,
            activations[l + 1]->rawData(m_begin, n_begin),
            activations[l + 1]->ld());

        if (it >= NWARMUP) {
          double dt = dsecnd() - t0;
          double flops = 2. * m * n * k;
#ifdef PRINT_PER_LAYER_PERFORMANCE
#pragma omp master
          {
            double gflops = flops / dt / 1e9;
            printf(
              "fwd layer %d %g ms %g GF/s %g GF/s/core\n",
              l, dt * 1e3, gflops, gflops / nthreads);
          }
#endif
          sum_times[tid][l][FWD] += dt;
#pragma omp master
          sum_flops[l][FWD] += flops;
        }
#pragma omp barrier
      } // for each layer

      // backward
      for (int l = nlayers - 1; l >= 0; --l) {
        // weight gradient computation
        double t0 = dsecnd();

        int m = nfeatures[l + 1], n = nfeatures[l], k = batch_size;
        int m_begin, m_end, n_begin, n_end;

#ifdef OVERLAP_ALL_REDUCE
        int nthreads_per_socket_for_gemm =
          nthreads_per_socket -
          nthreads_per_socket_for_allreduce[nthreads_per_socket];
#else
        int nthreads_per_socket_for_gemm = nthreads_per_socket;
#endif
        int tid_in_socket = get_thread_num_in_socket();
        if (tid_in_socket < nthreads_per_socket_for_gemm) {
          // partition k over socket
          int k_per_socket = (k + nsockets - 1) / nsockets;
          int k_begin = std::min(sid * k_per_socket, k);
          int k_end = std::min(k_begin + k_per_socket, k);

          // 2d partition m and n within socket
          // weight_grad gemm performs well with aspect ratio
          // 8 * (m_end - m_begin) ~= (n_end - n_begin)
          float aspect_ratio = 1./2;
          /*if (m == 512 && n == 1280) {
            aspect_ratio = 2.;
          }*/
          get_intra_socket_2dpartition(
              &m_begin,
              &m_end,
              &n_begin,
              &n_end,
              m,
              n,
              aspect_ratio,
              true /* m_align */,
              nthreads_per_socket_for_gemm,
              tid_in_socket);
          if (0 == it && 0 == tid) {
            int mb = (m + m_end - m_begin - 1) / (m_end - m_begin);
            int nb = (n + n_end - n_begin - 1) / (n_end - n_begin);
            int kb = k / nsockets / (k_end - k_begin);
            printf(
                "wgt m %d n %d k %d bm %d bn %d bk %d mb %d nb %d kb %d\n",
                m,
                n,
                k,
                m_end - m_begin,
                n_end - n_begin,
                k_end - k_begin,
                mb,
                nb,
                kb);
          }

          const float* A_begin = activations[l + 1]->rawData(k_begin, m_begin);
          cblas_sgemm(
              CblasRowMajor,
              CblasTrans,
              CblasNoTrans,
              m_end - m_begin,
              n_end - n_begin,
              k_end - k_begin,
              1.0f,
              A_begin,
              activations[l + 1]->ld(),
              activations[l]->rawData(k_begin, n_begin),
              activations[l]->ld(),
              0.0f,
              weight_grads[l]->rawData(sid * m + m_begin, n_begin),
              weight_grads[l]->ld());

          if (it >= NWARMUP) {
            double dt = dsecnd() - t0;
            double flops = 2. * m * n * k;
#ifdef PRINT_PER_LAYER_PERFORMANCE
#pragma omp master
            {
              double gflops = flops / dt / 1e9;
              printf(
                "wgt_gradient layer %d %g ms %g GF/s %g GF/s/core\n",
                l, dt * 1e3, gflops, gflops / nthreads);
            }
#endif
            sum_times[tid][l][WGT_GRAD] += dt;
#pragma omp master
            sum_flops[l][WGT_GRAD] += flops;
          }
        }

#pragma omp barrier

        // backward update
        t0 = dsecnd();

        m = batch_size, n = nfeatures[l], k = nfeatures[l + 1];

        if (tid_in_socket < nthreads_per_socket_for_gemm) {
          // backward gemm performs well with aspect ratio
          // (m_end - m_begin) ~= 32 * (n_end - n_begin)
          float aspect_ratio = 1./2;
          /*if (n == 512 && k == 512) {
            aspect_ratio = 4.;
          }
          else if (n == 1280 && k == 512) {
            aspect_ratio = 16.;
          }*/
          get_2dpartition(
              &m_begin,
              &m_end,
              &n_begin,
              &n_end,
              m,
              n,
              aspect_ratio,
              false /* m_align */,
              nthreads_per_socket_for_gemm,
              tid_in_socket);
          if (0 == it && 0 == tid) {
            int mb = (m / nsockets + m_end - m_begin - 1) / (m_end - m_begin);
            int nb = (n + n_end - n_begin - 1) / (n_end - n_begin);
            printf(
                "bwd m %d n %d k %d bm %d bn %d bk %d mb %d nb %d kb %d\n",
                m,
                n,
                k,
                m_end - m_begin,
                n_end - n_begin,
                k,
                mb,
                nb,
                1);
          }

          const float *A_begin = activations[l + 1]->rawData(m_begin, 0);
          Matrix<float, PAD> *C = activations[l == 0 ? nlayers + 1 : l].get();
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              m_end - m_begin,
              n_end - n_begin,
              k,
              1.0f,
              A_begin,
              activations[l + 1]->ld(),
              weights[l]->rawData(sid * k, n_begin),
              weights[l]->ld(),
              0.0f,
              C->rawData(m_begin, n_begin),
              C->ld());

          if (it >= NWARMUP) {
            double dt = dsecnd() - t0;
            double flops = 2. * m * n * k;
#ifdef PRINT_PER_LAYER_PERFORMANCE
#pragma omp master
            {
              double gflops = flops / dt / 1e9;
              printf(
                  "bwd layer %d %g ms %g GF/s %g GF/s/core\n",
                  l,
                  dt * 1e3,
                  gflops,
                  gflops / nthreads);
            }
#endif
            sum_times[tid][l][BWD] += dt;
#pragma omp master
            sum_flops[l][BWD] += flops;
          }
        }

#pragma omp barrier
      } // for each layer
    } // for each iteration
  } // omp parallel

  /////////////////////////////////////////////////////////////////////////////
  // compute load imbalance
  double load_imbalance[nlayers][NUM_BREAKDOWNS];
  double max_sum_times[nlayers][NUM_BREAKDOWNS];

  for (int l = 0; l < nlayers; ++l) {
    for (int i = FWD; i < NUM_BREAKDOWNS; ++i) {
      double sum = 0, max = 0;
      if (i == WGT_GRAD || i == BWD) {
        for (int sid = 0; sid < nsockets; ++sid) {
          for (int tid_in_socket = 0;
              tid_in_socket < nthreads_per_socket;
              ++tid_in_socket) {
            int tid = sid * nthreads_per_socket + tid_in_socket;
            sum += sum_times[tid][l][i];
            max = std::max(max, sum_times[tid][l][i]);
          }
        }
        max_sum_times[l][i] = max;

        double avg = sum / nthreads_per_socket / nsockets;
        load_imbalance[l][i] = max / avg;
      }
      else {
        for (int tid = 0; tid < nthreads; ++tid) {
          sum += sum_times[tid][l][i];
          max = std::max(max, sum_times[tid][l][i]);
        }
        max_sum_times[l][i] = max;

        double avg = sum / nthreads;
        load_imbalance[l][i] = max / avg;
      }
    }
  } // for each layer

  /////////////////////////////////////////////////////////////////////////////
  // report timing
  double
    total_times[NUM_BREAKDOWNS] = { 0 }, total_flops[NUM_BREAKDOWNS] = { 0 };
  for (int l = 0; l < nlayers; ++l) {
    printf(
      "[layer %d] fwd %g ms/iter (%g GF/s/core) imbalance %g, "
      "wgt_grad %g ms/iter (%g GF/s/core) imbalance %g, "
      "bwd %g ms/iter (%g GF/s/core) imbalance %g\n",
      l,
      max_sum_times[l][FWD] / NITER * 1e3,
      sum_flops[l][FWD] / max_sum_times[l][FWD] / nthreads / 1e9,
      load_imbalance[l][FWD],
      max_sum_times[l][WGT_GRAD] / NITER * 1e3,
      sum_flops[l][WGT_GRAD] / max_sum_times[l][WGT_GRAD] / nthreads / 1e9,
      load_imbalance[l][WGT_GRAD],
      max_sum_times[l][BWD] / NITER * 1e3,
      sum_flops[l][BWD] / max_sum_times[l][BWD] / nthreads / 1e9,
      load_imbalance[l][BWD]);

    for (int i = FWD; i < NUM_BREAKDOWNS; ++i) {
      total_times[i] += max_sum_times[l][i];
      total_flops[i] += sum_flops[l][i];
    }
  } // for each layer

  printf(
    "total fwd %g ms/iter (%g GF/s/core), "
    "wgt_grad %g ms/iter (%g GF/s/core), "
    "bwd %g ms/iter (%g GF/s/core)\n",
    total_times[FWD] / NITER * 1e3,
    total_flops[FWD] / total_times[FWD] / nthreads / 1e9,
    total_times[WGT_GRAD] / NITER * 1e3,
    total_flops[WGT_GRAD] / total_times[WGT_GRAD] / nthreads / 1e9,
    total_times[BWD] / NITER * 1e3,
    total_flops[BWD] / total_times[BWD] / nthreads / 1e9);

  /////////////////////////////////////////////////////////////////////////////
  // print check sum for correctness check
  for (int l = 0; l < nlayers; ++l) {
    double l1_norm = 0, l2_norm = 0, trace = 0;
    for (int i = 0; i < nfeatures[l + 1]; ++i) {
      for (int j = 0; j < nfeatures[l]; ++j) {
        float w = (*weights[l])(i, j);
        l1_norm += std::abs(w);
        l2_norm += w * w;
      }
      if (i < std::min(nfeatures[l + 1], nfeatures[l])) {
        trace += (*weights[l])(i, i);
      }
    }
    l2_norm = sqrt(l2_norm);
    printf("layer %d l1 %g l2 %g trace %g\n", l, l1_norm, l2_norm, trace);
  }

  return 0;
}
