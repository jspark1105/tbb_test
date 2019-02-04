#include "mlp_bench_common.h"

#include <cstring>

using namespace std;

double sum_times[MAX_NUM_THREADS][nlayers][NUM_BREAKDOWNS_ROUNDED_UP] = {0},
       sum_flops[nlayers][NUM_BREAKDOWNS] = {0};

unique_ptr<Matrix<float, PAD>> weights[nlayers], weight_grads[nlayers],
    weight_grad_push_bufs[nlayers], activations[nlayers + 2];

vector<array<int, 8>> rings = {{0, 1, 2, 4, 7, 6, 5, 3},
                               {0, 3, 5, 6, 7, 4, 2, 1}};

void init_matrices() {
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
}

double wall_clock_time = 0;
int nthreads, nthreads_per_socket;

void report_timing() {
  // Compute load imbalance
  double load_imbalance[nlayers][NUM_BREAKDOWNS];
  double max_sum_times[nlayers][NUM_BREAKDOWNS];

  for (int l = 0; l < nlayers; ++l) {
    for (int t = 0; t < nthreads; ++t) {
      sum_times[t][l][WGT_UPDATE] = sum_times[t][l][WGT_UPDATE_REDUCE_SCATTER] +
          sum_times[t][l][WGT_UPDATE_ALLGATHER];
    }
    sum_flops[l][WGT_UPDATE] = sum_flops[l][WGT_UPDATE_REDUCE_SCATTER] +
        sum_flops[l][WGT_UPDATE_ALLGATHER];
  }

  for (int l = 0; l < nlayers; ++l) {
    for (int i = FWD; i < NUM_BREAKDOWNS; ++i) {
      double sum = 0, max = 0;
      if (i == WGT_GRAD || i == BWD) {
        int nthreads_per_socket_for_gemm = nthreads_per_socket -
            nthreads_per_socket_for_allreduce[nthreads_per_socket];

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

        double avg = sum / nthreads_per_socket_for_gemm / nsockets;
        load_imbalance[l][i] = max / avg;
      } else {
        int nthreads_for_i = i >= WGT_UPDATE
            ? nthreads_per_socket_for_allreduce[nthreads_per_socket] * nsockets
            : nthreads;

        for (int tid = 0; tid < nthreads; ++tid) {
          sum += sum_times[tid][l][i];
          max = std::max(max, sum_times[tid][l][i]);
        }
        max_sum_times[l][i] = max;

        double avg = sum / nthreads_for_i;
        load_imbalance[l][i] = max / avg;
      }
    }
  } // for each layer

  // Report timing
  double total_times[NUM_BREAKDOWNS] = {0}, total_flops[NUM_BREAKDOWNS] = {0};
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
        sum_flops[l][WGT_UPDATE] / max_sum_times[l][WGT_UPDATE] / nsockets /
            1e9,
        load_imbalance[l][WGT_UPDATE],
        max_sum_times[l][WGT_UPDATE_REDUCE_SCATTER] / NITER * 1e3,
        sum_flops[l][WGT_UPDATE_REDUCE_SCATTER] /
            max_sum_times[l][WGT_UPDATE_REDUCE_SCATTER] / nsockets / 1e9,
        load_imbalance[l][WGT_UPDATE_REDUCE_SCATTER],
        max_sum_times[l][WGT_UPDATE_ALLGATHER] / NITER * 1e3,
        sum_flops[l][WGT_UPDATE_ALLGATHER] /
            max_sum_times[l][WGT_UPDATE_ALLGATHER] / nsockets / 1e9,
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
      total_flops[WGT_UPDATE_REDUCE_SCATTER] /
          total_times[WGT_UPDATE_REDUCE_SCATTER] / nsockets / 1e9,
      total_times[WGT_UPDATE_ALLGATHER] / NITER * 1e3,
      total_flops[WGT_UPDATE_ALLGATHER] / total_times[WGT_UPDATE_ALLGATHER] /
          nsockets / 1e9);
  printf("wall clock time %g ms/iter\n", wall_clock_time / NITER * 1e3);
}

void print_checksum() {
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
}

void get_my_ring_info(
    int sid,
    int task,
    int* idx_in_ring,
    int* prev_sid,
    int* next_sid) {
  int ring_to_use = task % rings.size();
  *idx_in_ring =
      std::find(rings[ring_to_use].begin(), rings[ring_to_use].end(), sid) -
      rings[ring_to_use].begin();
  *prev_sid = rings[ring_to_use][(*idx_in_ring - 1 + nsockets) % nsockets];
  *next_sid = rings[ring_to_use][(*idx_in_ring + 1) % nsockets];
  if (nsockets < 8) {
    *idx_in_ring = sid;
    *prev_sid = (sid - 1 + nsockets) % nsockets;
    *next_sid = (sid + 1) % nsockets;
    if (ring_to_use) {
      swap(*prev_sid, *next_sid);
    }
  }
}
