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
#include "TwistedHyperCube.h"

//#define PRINT_PER_LAYER_PERFORMANCE
//#define NO_ALL_REDUCE
#define USE_RING_ALL_REDUCE
//#define OVERLAP_ALL_REDUCE

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
  WGT_UPDATE,
  WGT_UPDATE_REDUCE_SCATTER,
  WGT_UPDATE_ALLGATHER,
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

volatile int done_flags[MAX_NUM_THREADS * 16] = { 0 };
volatile int ready_flags[MAX_NUM_THREADS * 16] = { 0 };

int nthreads_per_socket_for_allreduce[29] = {
  0,
  0,
  0,
  0,
  0,
  1, // total 5, 1 for all-reduce 4 for gemm
  1,
  1,
  1,
  1,
  2, // total 10, 2 for all-reduce 8 for gemm
  2,
  2,
  2,
  2,
  2,
  2,
  2,
  2,
  2,
  4, // total 20, 4 for all-reduce 16 for gemm
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
};

thread_local int done_flag_phase = 0;
thread_local int ready_flag_phase = 0;

static std::vector<std::array<int, 8>> rings = {
  { 0, 1, 2, 4, 7, 6, 5, 3 },
  { 0, 3, 5, 6, 7, 4, 2, 1 } };

void update_weight_reduce_scatter(
  Matrix<float, PAD> *weight_grad,
  array<Matrix<float, PAD> *, 3> weight_grad_push_buf,
  double alpha,
  int layer, bool measure_time,
  int nthreads_per_socket, int tid_in_socket)
{
  int nrows = weight_grad->nrows() / nsockets;
  int ncols = weight_grad->ncols();

  int sid = get_socket_num();
  int tid = nthreads_per_socket * sid + tid_in_socket;

  double t_reduce_scatter_begin = dsecnd();

#if defined(USE_RING_ALL_REDUCE)
  int ld = weight_grad->ld();
  size_t weight_size = nrows * ld;
  assert(weight_size % CACHE_LINE_LEN == 0);

  int ring_to_use = tid_in_socket % rings.size();

  int idx_in_ring =
    std::find(rings[ring_to_use].begin(), rings[ring_to_use].end(), sid) -
    rings[ring_to_use].begin();
  int prev_sid = rings[ring_to_use][(idx_in_ring - 1 + nsockets) % nsockets];
  int next_sid = rings[ring_to_use][(idx_in_ring + 1) % nsockets];
  if (nsockets < 8) {
    idx_in_ring = sid;
    prev_sid = (sid - 1 + nsockets) % nsockets;
    next_sid = (sid + 1) % nsockets;
    if (ring_to_use) {
      swap(prev_sid, next_sid);
    }
  }

  size_t i_per_chunk =
    (weight_size + nsockets * CACHE_LINE_LEN - 1)
    / nsockets / CACHE_LINE_LEN * CACHE_LINE_LEN;
  size_t i_per_thread =
    (i_per_chunk + nthreads_per_socket * CACHE_LINE_LEN - 1)
    / nthreads_per_socket / CACHE_LINE_LEN * CACHE_LINE_LEN;

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
    size_t thread_end =
      std::min(thread_begin + i_per_thread, chunk_end);

    size_t src_begin = socket_begin + thread_begin;
    size_t dst_begin = next_socket_begin + thread_begin;

    // push to buffer using non-temporal store
    for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
      _mm512_stream_si512(
        (__m512i *)(weight_grad_push_buf[0]->rawData() + dst_begin + i),
        _mm512_load_si512(weight_grad->rawData() + src_begin + i));
    }

    // we can proceed only when prev socket finished its push
    // need this fence to make sure stream stores have finished
    _mm_sfence();
    done_flags[tid * 16 + done_flag_phase] = 1;
    int flag_id_to_wait =
      (prev_sid * nthreads_per_socket + tid_in_socket) * 16 +
      done_flag_phase;
    while (!done_flags[flag_id_to_wait]) _mm_pause();
    done_flags[flag_id_to_wait] = 0;
    done_flag_phase = (done_flag_phase + 1) % 16;

    int chunk_to_read = (chunk_to_push - 1 + nsockets) % nsockets;
    chunk_begin = std::min(chunk_to_read * i_per_chunk, weight_size);
    chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    thread_begin =
      std::min(chunk_begin + tid_in_socket * i_per_thread, chunk_end);
    thread_end =
      std::min(thread_begin + i_per_thread, chunk_end);

    dst_begin = socket_begin + thread_begin;

    // accumulate wgt grads
    if (step < nsockets - 2) {
      for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
        _mm512_stream_ps(
          weight_grad->rawData() + dst_begin + i,
          _mm512_add_ps(
            _mm512_load_ps(weight_grad->rawData() + dst_begin + i),
            _mm512_load_ps(
              weight_grad_push_buf[0]->rawData() + dst_begin + i)));
      }
    }
  }

#else
  // !USE_RING_ALL_REDUCE -> meaning naive all reduce
  pair<int, int> i_range = get_partition(
    nrows, nthreads_per_socket, tid_in_socket);
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
  Matrix<float, PAD> *weight,
  Matrix<float, PAD> *weight_grad,
  array<Matrix<float, PAD> *, 3> weight_grad_push_buf,
  double alpha,
  int layer, bool measure_time,
  int nthreads_per_socket, int tid_in_socket)
{
  int nrows = weight->nrows() / nsockets;
  int ncols = weight->ncols();

  int sid = get_socket_num();
  int tid = nthreads_per_socket * sid + tid_in_socket;

  double t_allgather_begin = dsecnd();

#if defined(USE_RING_ALL_REDUCE)
  int ld = weight_grad->ld();
  size_t weight_size = nrows * ld;
  assert(weight_size % CACHE_LINE_LEN == 0);

  int ring_to_use = tid_in_socket % rings.size();

  int idx_in_ring =
    std::find(rings[ring_to_use].begin(), rings[ring_to_use].end(), sid) -
    rings[ring_to_use].begin();
  int prev_sid = rings[ring_to_use][(idx_in_ring - 1 + nsockets) % nsockets];
  int next_sid = rings[ring_to_use][(idx_in_ring + 1) % nsockets];
  if (nsockets < 8) {
    idx_in_ring = sid;
    prev_sid = (sid - 1 + nsockets) % nsockets;
    next_sid = (sid + 1) % nsockets;
    if (ring_to_use) {
      swap(prev_sid, next_sid);
    }
  }

  size_t i_per_chunk =
    (weight_size + nsockets * CACHE_LINE_LEN - 1)
    / nsockets / CACHE_LINE_LEN * CACHE_LINE_LEN;
  size_t i_per_thread =
    (i_per_chunk + nthreads_per_socket * CACHE_LINE_LEN - 1)
    / nthreads_per_socket / CACHE_LINE_LEN * CACHE_LINE_LEN;

  size_t socket_begin = sid * weight_size;
  size_t next_socket_begin = next_sid * weight_size;

  if (nsockets == 1) {
    int chunk_to_push = 0;
    size_t chunk_begin = std::min(chunk_to_push * i_per_chunk, weight_size);
    size_t chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    size_t thread_begin =
      std::min(chunk_begin + tid_in_socket * i_per_thread, chunk_end);
    size_t thread_end =
      std::min(thread_begin + i_per_thread, chunk_end);

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
    size_t thread_end =
      std::min(thread_begin + i_per_thread, chunk_end);

    size_t src_begin = socket_begin + thread_begin;
    size_t dst_begin = next_socket_begin + thread_begin;

    // add reduced wgt grad to wgt
    if (0 == step) {
      for (size_t i = 0; i < thread_end - thread_begin; i += CACHE_LINE_LEN) {
        __m512 temp_v = _mm512_add_ps(
          _mm512_load_ps(weight_grad->rawData() + src_begin + i),
          _mm512_load_ps(weight_grad_push_buf[0]->rawData() + src_begin + i));

        temp_v = _mm512_fmadd_ps(
          _mm512_set1_ps(-alpha),
          temp_v,
          _mm512_load_ps(weight->rawData() + src_begin + i));
        _mm512_store_ps(weight->rawData() + src_begin + i, temp_v);
        _mm512_stream_ps(weight->rawData() + dst_begin + i, temp_v);
      }
    }
    else {
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
      (prev_sid * nthreads_per_socket + tid_in_socket) * 16 +
      done_flag_phase;
    while (!done_flags[flag_id_to_wait]) _mm_pause();
    done_flags[flag_id_to_wait] = 0;
    done_flag_phase = (done_flag_phase + 1) % 16;
  }
#else
  // !USE_RING_ALL_REDUCE
  pair<int, int> i_range = get_partition(
    nrows, nthreads_per_socket, tid_in_socket);
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
  Matrix<float, PAD> *weight,
  Matrix<float, PAD> *weight_grad,
  array<Matrix<float, PAD> *, 3> weight_grad_push_buf,
  double alpha,
  int layer, bool measure_time,
  int nthreads_per_socket, int tid_in_socket) {
  update_weight_reduce_scatter(
    weight_grad, weight_grad_push_buf, alpha, layer, measure_time,
    nthreads_per_socket, tid_in_socket);
  update_weight_allgather(
    weight, weight_grad, weight_grad_push_buf, alpha, layer, measure_time,
    nthreads_per_socket, tid_in_socket);
}

void check_all_reduce_correctness(
  Matrix<float, PAD> *weight,
  Matrix<float, PAD> *weight_grad,
  array<Matrix<float, PAD> *, 3> weight_grad_push_buf)
{
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
    weight, weight_grad, weight_grad_push_buf, 1, 0, false,
    get_num_threads_per_socket(), get_thread_num_in_socket());

  for (int sid = 0; sid < nsockets; ++sid) {
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        float expected =
          i * j -
          ((i + j) * nsockets + (nsockets - 1) * nsockets / 2);
        float actual = (*weight)(sid * nrows + i, j);
        float abs_err = std::abs(actual - expected);
        if (abs_err > 1e-5) {
          printf(
            "sid %d i %d j %d expected %f actual %f\n",
            sid, i, j, expected, actual);
          exit(-1);
        }
      }
    }
  }
}

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
    weights[nlayers], weight_grads[nlayers], weight_grad_push_bufs[nlayers][3],
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

    for (int i = 0; i < 3; ++i) {
      weight_grad_push_bufs[l][i].reset(
        create_matrix_with_numa_aware_allocation(
          nsockets * nfeatures[l + 1], nfeatures[l]));
    }
  }
  activations[nlayers].reset(
    create_matrix_with_numa_aware_allocation(batch_size, nfeatures[nlayers]));
  // this extra is to avoid overwriting the input
  activations[nlayers + 1].reset(
    create_matrix_with_numa_aware_allocation(batch_size, nfeatures[0]));

  /////////////////////////////////////////////////////////////////////////////
  // check correctness of all-reduce
  check_all_reduce_correctness(
    weights[0].get(), weight_grads[0].get(),
    { weight_grad_push_bufs[0][0].get(),
      weight_grad_push_bufs[0][1].get(),
      weight_grad_push_bufs[0][2].get()});

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
        else if (l < nlayers - 1) {
#ifndef NO_ALL_REDUCE
          int tid_in_socket_for_allreduce =
            tid_in_socket - nthreads_per_socket_for_gemm;
          update_weight_allgather(
            weights[l + 1].get(),
            weight_grads[l + 1].get(),
            { weight_grad_push_bufs[l + 1][0].get(),
              weight_grad_push_bufs[l + 1][1].get(),
              weight_grad_push_bufs[l + 1][2].get() },
            1e-10,
            l + 1, it >= NWARMUP,
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
        else {
#ifndef NO_ALL_REDUCE
          int tid_in_socket_for_allreduce =
            tid_in_socket - nthreads_per_socket_for_gemm;
          update_weight_reduce_scatter(
            weight_grads[l].get(),
            { weight_grad_push_bufs[l][0].get(),
              weight_grad_push_bufs[l][1].get(),
              weight_grad_push_bufs[l][2].get() },
            1e-10,
            l, it >= NWARMUP,
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
            { weight_grad_push_bufs[l][0].get(),
              weight_grad_push_bufs[l][1].get(),
              weight_grad_push_bufs[l][2].get() },
            1e-10,
            l, it >= NWARMUP,
            nthreads_per_socket_for_allreduce[nthreads_per_socket],
            tid_in_socket_for_allreduce);
        }
#else
        update_weight(
            weights[l].get(),
            weight_grads[l].get(),
            { weight_grad_push_bufs[l][0].get(),
              weight_grad_push_bufs[l][1].get(),
              weight_grad_push_bufs[l][2].get() },
            1e-10,
            l, it >= NWARMUP,
            nthreads_per_socket,
            tid_in_socket);
#endif
#endif // !NO_ALL_REDUCE
      } // for each layer
    } // for each iteration
  } // omp parallel

  /////////////////////////////////////////////////////////////////////////////
  // compute load imbalance
  double load_imbalance[nlayers][NUM_BREAKDOWNS];
  double max_sum_times[nlayers][NUM_BREAKDOWNS];

  for (int l = 0; l < nlayers; ++l) {
    for (int t = 0; t < nthreads; ++t) {
      sum_times[t][l][WGT_UPDATE] =
        sum_times[t][l][WGT_UPDATE_REDUCE_SCATTER] +
        sum_times[t][l][WGT_UPDATE_ALLGATHER];
    }
    sum_flops[l][WGT_UPDATE] =
      sum_flops[l][WGT_UPDATE_REDUCE_SCATTER] +
      sum_flops[l][WGT_UPDATE_ALLGATHER];
  }

  for (int l = 0; l < nlayers; ++l) {
    for (int i = FWD; i < NUM_BREAKDOWNS; ++i) {
      double sum = 0, max = 0;
      if (i == WGT_GRAD || i == BWD) {
        int nthreads_per_socket_for_gemm =
          nthreads_per_socket -
          nthreads_per_socket_for_allreduce[nthreads_per_socket];

        for (int sid = 0; sid < nsockets; ++sid) {
          for (int tid_in_socket = 0;
              tid_in_socket < nthreads_per_socket_for_gemm;
              ++tid_in_socket) {
            int tid = sid * nthreads_per_socket + tid_in_socket;
            sum += sum_times[tid][l][i];
            max = std::max(max, sum_times[tid][l][i]);
          }
        }
        max_sum_times[l][i] = max;

        double avg = sum / nthreads_per_socket_for_gemm / nsockets;
        load_imbalance[l][i] = max / avg;
      }
      else {
        int nthreads_for_i =
          i >= WGT_UPDATE
          ? nthreads_per_socket_for_allreduce[nthreads_per_socket] * nsockets
          : nthreads;

        for (int tid = 0; tid < nthreads_for_i; ++tid) {
          sum += sum_times[tid][l][i];
          max = std::max(max, sum_times[tid][l][i]);
        }
        max_sum_times[l][i] = max;

        double avg = sum / nthreads_for_i;
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
      "bwd %g ms/iter (%g GF/s/core) imbalance %g, "
      "wgt_update %g ms/iter (%g GB/s/socket) imbalance %g, "
      "wgt_update_reduce_scatter %g ms/iter (%g GB/s/socket) imbalance %g, "
      "wgt_update_allgather %g ms/iter (%g GB/s/socket) imbalance %g\n",
      l,
      max_sum_times[l][FWD] / NITER * 1e3,
      sum_flops[l][FWD] / max_sum_times[l][FWD] / nthreads / 1e9,
      load_imbalance[l][FWD],
      max_sum_times[l][WGT_GRAD] / NITER * 1e3,
      sum_flops[l][WGT_GRAD] / max_sum_times[l][WGT_GRAD] / nthreads / 1e9,
      load_imbalance[l][WGT_GRAD],
      max_sum_times[l][BWD] / NITER * 1e3,
      sum_flops[l][BWD] / max_sum_times[l][BWD] / nthreads / 1e9,
      load_imbalance[l][BWD],
      max_sum_times[l][WGT_UPDATE] / NITER * 1e3,
      sum_flops[l][WGT_UPDATE] / max_sum_times[l][WGT_UPDATE] / nsockets / 1e9,
      load_imbalance[l][WGT_UPDATE],
      max_sum_times[l][WGT_UPDATE_REDUCE_SCATTER] / NITER * 1e3,
      sum_flops[l][WGT_UPDATE_REDUCE_SCATTER] / max_sum_times[l][WGT_UPDATE_REDUCE_SCATTER] / nsockets / 1e9,
      load_imbalance[l][WGT_UPDATE_REDUCE_SCATTER],
      max_sum_times[l][WGT_UPDATE_ALLGATHER] / NITER * 1e3,
      sum_flops[l][WGT_UPDATE_ALLGATHER] / max_sum_times[l][WGT_UPDATE_ALLGATHER] / nsockets / 1e9,
      load_imbalance[l][WGT_UPDATE_ALLGATHER]);

    for (int i = FWD; i < NUM_BREAKDOWNS; ++i) {
      total_times[i] += max_sum_times[l][i];
      total_flops[i] += sum_flops[l][i];
    }
  } // for each layer

  printf(
    "total fwd %g ms/iter (%g GF/s/core), "
    "wgt_grad %g ms/iter (%g GF/s/core), "
    "bwd %g ms/iter (%g GF/s/core), "
    "wgt_update %g ms/iter (%g GB/s/socket), "
    "wgt_update_reduce_scatter %g ms/iter (%g GB/s/socket), "
    "wgt_update_allgather %g ms/iter (%g GB/s/socket)\n",
    total_times[FWD] / NITER * 1e3,
    total_flops[FWD] / total_times[FWD] / nthreads / 1e9,
    total_times[WGT_GRAD] / NITER * 1e3,
    total_flops[WGT_GRAD] / total_times[WGT_GRAD] / nthreads / 1e9,
    total_times[BWD] / NITER * 1e3,
    total_flops[BWD] / total_times[BWD] / nthreads / 1e9,
    total_times[WGT_UPDATE] / NITER * 1e3,
    total_flops[WGT_UPDATE] / total_times[WGT_UPDATE] / nsockets / 1e9,
    total_times[WGT_UPDATE_REDUCE_SCATTER] / NITER * 1e3,
    total_flops[WGT_UPDATE_REDUCE_SCATTER] / total_times[WGT_UPDATE_REDUCE_SCATTER] / nsockets / 1e9,
    total_times[WGT_UPDATE_ALLGATHER] / NITER * 1e3,
    total_flops[WGT_UPDATE_ALLGATHER] / total_times[WGT_UPDATE_ALLGATHER] / nsockets / 1e9);

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
