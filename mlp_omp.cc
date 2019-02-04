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
#include <immintrin.h>

#include "Matrix.h"
#include "Partition.h"
#include "Rand.h"
#include "TwistedHyperCube.h"
#include "mlp_bench_common.h"

//#define PRINT_PER_LAYER_PERFORMANCE
//#define NO_ALL_REDUCE
#define USE_RING_ALL_REDUCE
#define OVERLAP_ALL_REDUCE

using namespace std;

Matrix<float, PAD>* create_matrix_with_numa_aware_allocation(
    int nrows,
    int ncols) {
  Matrix<float, PAD>* matrix = new Matrix<float, PAD>(nrows, ncols);
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

volatile int done_flags[MAX_NUM_THREADS * 16] = {0};
volatile int ready_flags[MAX_NUM_THREADS * 16] = {0};

thread_local int done_flag_phase = 0;
thread_local int ready_flag_phase = 0;

void update_weight_reduce_scatter(
    Matrix<float, PAD>* weight_grad,
    Matrix<float, PAD>* weight_grad_push_buf,
    double alpha,
    int layer,
    bool measure_time,
    int nthreads_per_socket,
    int tid_in_socket) {
  int nrows = weight_grad->nrows() / nsockets;
  int ncols = weight_grad->ncols();

  int sid = get_socket_num();
  int tid = nthreads_per_socket * sid + tid_in_socket;

  double t_reduce_scatter_begin = dsecnd();

#if defined(USE_RING_ALL_REDUCE)
  int ld = weight_grad->ld();
  size_t weight_size = nrows * ld;
  assert(weight_size % CACHE_LINE_LEN == 0);

  int idx_in_ring, prev_sid, next_sid;
  get_my_ring_info(sid, tid_in_socket, &idx_in_ring, &prev_sid, &next_sid);

  size_t i_per_chunk = (weight_size + nsockets * CACHE_LINE_LEN - 1) /
      nsockets / CACHE_LINE_LEN * CACHE_LINE_LEN;
  size_t i_per_thread =
      (i_per_chunk + nthreads_per_socket * CACHE_LINE_LEN - 1) /
      nthreads_per_socket / CACHE_LINE_LEN * CACHE_LINE_LEN;

  size_t socket_begin = sid * weight_size;
  size_t next_socket_begin = next_sid * weight_size;

  // reduce-scatter phase
  for (int step = 0; step < nsockets - 1; ++step) {
    // we partition the array into nsockets chunks
    // at ith step, socket s reads (nsockets - 1 + s - i)th chunk from
    // socket s - 1 and accumulates to its local chunk
    int chunk_to_push = (idx_in_ring - step + nsockets) % nsockets;

    size_t chunk_begin = std::min(chunk_to_push * i_per_chunk, weight_size);
    size_t chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    size_t thread_begin =
        std::min(chunk_begin + tid_in_socket * i_per_thread, chunk_end);
    size_t thread_end = std::min(thread_begin + i_per_thread, chunk_end);

    size_t src_begin = socket_begin + thread_begin;
    size_t dst_begin = next_socket_begin + thread_begin;

    // push to buffer using non-temporal store
    for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
      _mm512_stream_si512(
          reinterpret_cast<__m512i*>(
              weight_grad_push_buf->rawData() + dst_begin + i),
          _mm512_load_si512(weight_grad->rawData() + src_begin + i));
    }

    // we can proceed only when prev socket finished its push
    // need this fence to make sure stream stores have finished
    _mm_sfence();
    done_flags[tid * 16 + done_flag_phase] = 1;
    int flag_id_to_wait =
        (prev_sid * nthreads_per_socket + tid_in_socket) * 16 + done_flag_phase;
    while (!done_flags[flag_id_to_wait])
      _mm_pause();
    done_flags[flag_id_to_wait] = 0;
    done_flag_phase = (done_flag_phase + 1) % 16;

    int chunk_to_read = (chunk_to_push - 1 + nsockets) % nsockets;
    chunk_begin = std::min(chunk_to_read * i_per_chunk, weight_size);
    chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    thread_begin =
        std::min(chunk_begin + tid_in_socket * i_per_thread, chunk_end);
    thread_end = std::min(thread_begin + i_per_thread, chunk_end);

    dst_begin = socket_begin + thread_begin;

    // accumulate wgt grads
    if (step < nsockets - 2) {
      for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
        _mm512_stream_ps(
            weight_grad->rawData() + dst_begin + i,
            _mm512_add_ps(
                _mm512_load_ps(weight_grad->rawData() + dst_begin + i),
                _mm512_load_ps(
                    weight_grad_push_buf->rawData() + dst_begin + i)));
      }
    }
  }

#else
  // !USE_RING_ALL_REDUCE -> meaning naive all reduce
  pair<int, int> i_range =
      get_partition(nrows, nthreads_per_socket, tid_in_socket);
  for (int i = i_range.first; i < i_range.second; ++i) {
    for (int j = 0; j < ncols; ++j) {
      for (int s = 1; s < nsockets; ++s) {
        (*weight_grad)(i, j) += (*weight_grad)(s * nrows + i, j);
      }
    }
  }
#endif // !USE_RING_ALL_REDUCE

  if (measure_time) {
    double dt = dsecnd() - t_reduce_scatter_begin;
    double bytes = (nsockets - 1) * nrows * ncols * sizeof(float);
    sum_times[tid][layer][WGT_UPDATE_REDUCE_SCATTER] += dt;
    if (tid_in_socket == 0 && sid == 0) {
      sum_flops[layer][WGT_UPDATE_REDUCE_SCATTER] += bytes;
    }
  }
}

void update_weight_allgather(
    Matrix<float, PAD>* weight,
    Matrix<float, PAD>* weight_grad,
    Matrix<float, PAD>* weight_grad_push_buf,
    double alpha,
    int layer,
    bool measure_time,
    int nthreads_per_socket,
    int tid_in_socket) {
  int nrows = weight->nrows() / nsockets;
  int ncols = weight->ncols();

  int sid = get_socket_num();
  int tid = nthreads_per_socket * sid + tid_in_socket;

  double t_allgather_begin = dsecnd();

#if defined(USE_RING_ALL_REDUCE)
  int ld = weight_grad->ld();
  size_t weight_size = nrows * ld;
  assert(weight_size % CACHE_LINE_LEN == 0);

  int idx_in_ring, prev_sid, next_sid;
  get_my_ring_info(sid, tid_in_socket, &idx_in_ring, &prev_sid, &next_sid);

  size_t i_per_chunk = (weight_size + nsockets * CACHE_LINE_LEN - 1) /
      nsockets / CACHE_LINE_LEN * CACHE_LINE_LEN;
  size_t i_per_thread =
      (i_per_chunk + nthreads_per_socket * CACHE_LINE_LEN - 1) /
      nthreads_per_socket / CACHE_LINE_LEN * CACHE_LINE_LEN;

  size_t socket_begin = sid * weight_size;
  size_t next_socket_begin = next_sid * weight_size;

  if (nsockets == 1) {
    int chunk_to_push = 0;
    size_t chunk_begin = std::min(chunk_to_push * i_per_chunk, weight_size);
    size_t chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    size_t thread_begin =
        std::min(chunk_begin + tid_in_socket * i_per_thread, chunk_end);
    size_t thread_end = std::min(thread_begin + i_per_thread, chunk_end);

    size_t src_begin = socket_begin + thread_begin;

    // add reduced wgt grad to wgt
    for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
      __m512 temp_v = _mm512_fmadd_ps(
          _mm512_set1_ps(-alpha),
          _mm512_load_ps(weight_grad->rawData() + src_begin + i),
          _mm512_load_ps(weight->rawData() + src_begin + i));
      _mm512_store_ps(weight->rawData() + src_begin + i, temp_v);
    }
  }

  // allgather phase
  for (int step = 0; step < (int)nsockets - 1; ++step) {
    int chunk_to_push = (idx_in_ring + 1 - step + nsockets) % nsockets;
    size_t chunk_begin = std::min(chunk_to_push * i_per_chunk, weight_size);
    size_t chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    size_t thread_begin =
        std::min(chunk_begin + tid_in_socket * i_per_thread, chunk_end);
    size_t thread_end = std::min(thread_begin + i_per_thread, chunk_end);

    size_t src_begin = socket_begin + thread_begin;
    size_t dst_begin = next_socket_begin + thread_begin;

    // add reduced wgt grad to wgt
    if (0 == step) {
      for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
        __m512 temp_v = _mm512_add_ps(
            _mm512_load_ps(weight_grad->rawData() + src_begin + i),
            _mm512_load_ps(weight_grad_push_buf->rawData() + src_begin + i));

        temp_v = _mm512_fmadd_ps(
            _mm512_set1_ps(-alpha),
            temp_v,
            _mm512_load_ps(weight->rawData() + src_begin + i));
        _mm512_store_ps(weight->rawData() + src_begin + i, temp_v);
        _mm512_stream_ps(weight->rawData() + dst_begin + i, temp_v);
      }
    } else {
      for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
        _mm512_stream_ps(
            weight->rawData() + dst_begin + i,
            _mm512_load_ps(weight->rawData() + src_begin + i));
      }
    }

    // we can proceed only when prev socket finished its push
    // need this fence to make sure stream stores have finished
    _mm_sfence();
    done_flags[tid * 16 + done_flag_phase] = 1;
    int flag_id_to_wait =
        (prev_sid * nthreads_per_socket + tid_in_socket) * 16 + done_flag_phase;
    while (!done_flags[flag_id_to_wait])
      _mm_pause();
    done_flags[flag_id_to_wait] = 0;
    done_flag_phase = (done_flag_phase + 1) % 16;
  }
#else
  // !USE_RING_ALL_REDUCE
  pair<int, int> i_range =
      get_partition(nrows, nthreads_per_socket, tid_in_socket);
  for (int i = i_range.first; i < i_range.second; ++i) {
    for (int j = 0; j < ncols; ++j) {
      (*weight)(i, j) -= alpha * (*weight_grad)(i, j);

      for (int s = 1; s < nsockets; ++s) {
        (*weight)(s * nrows + i, j) = (*weight)(i, j);
      }
    }
  }
#endif // !USE_RING_ALL_REDUCE

  if (measure_time) {
    double bytes = (nsockets - 1) * nrows * ncols * sizeof(float);
    double dt = dsecnd() - t_allgather_begin;
    sum_times[tid][layer][WGT_UPDATE_ALLGATHER] += dt;
    if (tid_in_socket == 0 && sid == 0) {
      sum_flops[layer][WGT_UPDATE_ALLGATHER] += bytes;
    }
  }
}

void update_weight(
    Matrix<float, PAD>* weight,
    Matrix<float, PAD>* weight_grad,
    Matrix<float, PAD>* weight_grad_push_buf,
    double alpha,
    int layer,
    bool measure_time,
    int nthreads_per_socket,
    int tid_in_socket) {
  update_weight_reduce_scatter(
      weight_grad,
      weight_grad_push_buf,
      alpha,
      layer,
      measure_time,
      nthreads_per_socket,
      tid_in_socket);
  update_weight_allgather(
      weight,
      weight_grad,
      weight_grad_push_buf,
      alpha,
      layer,
      measure_time,
      nthreads_per_socket,
      tid_in_socket);
}

void check_all_reduce_correctness(
    Matrix<float, PAD>* weight,
    Matrix<float, PAD>* weight_grad,
    Matrix<float, PAD>* weight_grad_push_buf) {
  int nrows = weight->nrows() / nsockets;
  int ncols = weight->ncols();

#pragma omp parallel for collapse(2)
  for (int sid = 0; sid < nsockets; ++sid) {
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        (*weight_grad)(sid * nrows + i, j) = sid + i + j;
        (*weight)(sid * nrows + i, j) = i * j;
      }
    }
  }

#pragma omp parallel
  update_weight(
      weight,
      weight_grad,
      weight_grad_push_buf,
      1,
      0,
      false,
      get_num_threads_per_socket(),
      get_thread_num_in_socket());

  for (int sid = 0; sid < nsockets; ++sid) {
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        float expected =
            i * j - ((i + j) * nsockets + (nsockets - 1) * nsockets / 2);
        float actual = (*weight)(sid * nrows + i, j);
        float abs_err = std::abs(actual - expected);
        if (abs_err > 1e-5) {
          printf(
              "sid %d i %d j %d expected %f actual %f\n",
              sid,
              i,
              j,
              expected,
              actual);
          exit(-1);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "%s <nsockets> <nthreads_per_socket>\n", argv[0]);
    exit(1);
  }

  nsockets = atoi(argv[1]);
  nthreads_per_socket = atoi(argv[2]);
  nthreads = nsockets * nthreads_per_socket;
  omp_set_num_threads(nthreads);

  int batch_size = 1024 * nsockets; // weak-scaling with nsockets

  /////////////////////////////////////////////////////////////////////////////
  // allocate memory and "first-touch" for NUMA-aware allocation
  for (int l = 0; l < nlayers; ++l) {
    activations[l].reset(
        create_matrix_with_numa_aware_allocation(batch_size, nfeatures[l]));

    // Weights and their gradients are replicated at each socket.
    weights[l].reset(create_matrix_with_numa_aware_allocation(
        nsockets * nfeatures[l + 1], nfeatures[l]));

    weight_grads[l].reset(create_matrix_with_numa_aware_allocation(
        nsockets * nfeatures[l + 1], nfeatures[l]));

    weight_grad_push_bufs[l].reset(
        create_matrix_with_numa_aware_allocation(
            nsockets * nfeatures[l + 1], nfeatures[l]));
  }
  activations[nlayers].reset(
      create_matrix_with_numa_aware_allocation(batch_size, nfeatures[nlayers]));
  // this extra is to avoid overwriting the input
  activations[nlayers + 1].reset(
    create_matrix_with_numa_aware_allocation(batch_size, nfeatures[0]));

  /////////////////////////////////////////////////////////////////////////////
  // check correctness of all-reduce
  check_all_reduce_correctness(
      weights[0].get(),
      weight_grads[0].get(),
      weight_grad_push_bufs[0].get());

  /////////////////////////////////////////////////////////////////////////////
  // initialize values (only done by the master thread to be deterministic)
  init_matrices();

  /////////////////////////////////////////////////////////////////////////////
  // Main computation
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
      if (it == NWARMUP && tid == 0) {
        wall_clock_time = dsecnd();
      }

      // forward
      for (int l = 0; l < nlayers; ++l) {
        double t0 = dsecnd();

        int m = batch_size, n = nfeatures[l + 1], k = nfeatures[l];
        int m_begin, m_end, n_begin, n_end;
        // forward gemm performs well with aspect ratio
        // (m_end - m_begin) ~= 32 * (n_end - n_begin)
        float aspect_ratio = 4.;
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
                l,
                dt * 1e3,
                gflops,
                gflops / nthreads);
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
        int nthreads_per_socket_for_gemm = nthreads_per_socket -
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
          float aspect_ratio = 1. / 2;
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
                  l,
                  dt * 1e3,
                  gflops,
                  gflops / nthreads);
            }
#endif
            sum_times[tid][l][WGT_GRAD] += dt;
#pragma omp master
            sum_flops[l][WGT_GRAD] += flops;
          }
        } else if (l < nlayers - 1) {
#ifndef NO_ALL_REDUCE
          int tid_in_socket_for_allreduce =
              tid_in_socket - nthreads_per_socket_for_gemm;
          update_weight_allgather(
              weights[l + 1].get(),
              weight_grads[l + 1].get(),
              weight_grad_push_bufs[l + 1].get(),
              1e-10,
              l + 1,
              it >= NWARMUP,
              nthreads_per_socket_for_allreduce[nthreads_per_socket],
              tid_in_socket_for_allreduce);
#endif // !NO_ALL_REDUCE
        }

#pragma omp barrier

        // backward update
        t0 = dsecnd();

        m = batch_size, n = nfeatures[l], k = nfeatures[l + 1];

        if (tid_in_socket < nthreads_per_socket_for_gemm) {
          // backward gemm performs well with aspect ratio
          // (m_end - m_begin) ~= 32 * (n_end - n_begin)
          float aspect_ratio = 1. / 2;
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

          const float* A_begin = activations[l + 1]->rawData(m_begin, 0);
          Matrix<float, PAD>* C = activations[l == 0 ? nlayers + 1 : l].get();
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
        } else {
#ifndef NO_ALL_REDUCE
          int tid_in_socket_for_allreduce =
              tid_in_socket - nthreads_per_socket_for_gemm;
          update_weight_reduce_scatter(
              weight_grads[l].get(),
              weight_grad_push_bufs[l].get(),
              1e-10,
              l,
              it >= NWARMUP,
              nthreads_per_socket_for_allreduce[nthreads_per_socket],
              tid_in_socket_for_allreduce);
#endif
        }

#pragma omp barrier

#ifndef NO_ALL_REDUCE
#ifdef OVERLAP_ALL_REDUCE
        if (l == 0 && tid_in_socket >= nthreads_per_socket_for_gemm) {
          int tid_in_socket_for_allreduce =
              tid_in_socket - nthreads_per_socket_for_gemm;
          update_weight_allgather(
              weights[l].get(),
              weight_grads[l].get(),
              weight_grad_push_bufs[l].get(),
              1e-10,
              l,
              it >= NWARMUP,
              nthreads_per_socket_for_allreduce[nthreads_per_socket],
              tid_in_socket_for_allreduce);
        }
#else
        update_weight(
            weights[l].get(),
            weight_grads[l].get(),
            weight_grad_push_bufs[l].get(),
            1e-10,
            l,
            it >= NWARMUP,
            nthreads_per_socket,
            tid_in_socket);
#endif
#endif // !NO_ALL_REDUCE
      } // for each layer
    } // for each iteration
  } // omp parallel
  wall_clock_time = dsecnd() - wall_clock_time;

  /////////////////////////////////////////////////////////////////////////////
  // report timing
  report_timing();

  /////////////////////////////////////////////////////////////////////////////
  // print check sum for correctness check
  print_checksum();

  return 0;
}
