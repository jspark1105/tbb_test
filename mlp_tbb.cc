#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include <mkl.h>
#include <numa.h>
#include <omp.h>
#include <x86intrin.h>
// #include <glog/logging.h>

#define __TBB_PREVIEW_LIGHTWEIGHT_POLICY 1
#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include <tbb/task_scheduler_init.h>
#include <tbb/task_scheduler_observer.h>

#include "Matrix.h"
#include "Partition.h"
#include "Rand.h"

// #define PRINT_PER_LAYER_PERFORMANCE

using namespace std;
namespace flow = tbb::flow;

constexpr int PAD = 16;
constexpr int CACHE_LINE_LEN = 16;

int nthreads_per_socket;
int nthreads;

class pinning_observer : public tbb::task_scheduler_observer {
 public:
  pinning_observer(tbb::task_arena& arena, int numa_node_id)
      : tbb::task_scheduler_observer(arena), numa_node_id_(numa_node_id) {
    observe(true);
  } // activate the observer

  void on_scheduler_entry(bool /* unused */) override {
    auto bm = numa_allocate_nodemask();
    numa_bitmask_clearall(bm);
    numa_bitmask_setbit(bm, numa_node_id_);
    numa_bind(bm);
    numa_bitmask_free(bm);
    if (numa_node_id_ != -1) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      int tid = tbb::task_arena::current_thread_index();
      int ncores_per_socket =
          std::thread::hardware_concurrency() / numa_num_configured_nodes() / 2;
      CPU_SET(tid + numa_node_id_ * ncores_per_socket, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    } else {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) {
        CPU_SET(i, &cpuset);
      }
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
  }

 private:
  int numa_node_id_;
};

vector<unique_ptr<tbb::task_arena>> tbb_arena;
vector<unique_ptr<pinning_observer>> tbb_observers;

Matrix<float, PAD>* create_matrix_with_numa_aware_allocation(
    int nrows,
    int ncols) {
  Matrix<float, PAD>* matrix = new Matrix<float, PAD>(nrows, ncols);

  for (int sid = 0; sid < nsockets; ++sid) {
    tbb_arena[sid]->execute([&]() {
      tbb::parallel_for(
          0,
          nthreads_per_socket,
          [&](size_t task_id) {
            pair<int, int> i_range =
                get_partition(nrows, sid, nthreads_per_socket, task_id);
            for (int i = i_range.first; i < i_range.second; ++i) {
              for (int j = 0; j < ncols; ++j) {
                (*matrix)(i, j) = (i + j) % 31 - 15;
              }
            }
          },
          tbb::simple_partitioner());
    });
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
int nfeatures[] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1};
// int nfeatures[] = { 512, 512 };
constexpr int nlayers = sizeof(nfeatures) / sizeof(nfeatures[0]) - 1;
constexpr int MAX_NUM_THREADS = 1024;
constexpr int NWARMUP = 2, NITER = 256;

// Being careful not to have false sharing
constexpr int NUM_BREAKDOWNS_ROUNDED_UP =
    (NUM_BREAKDOWNS + CACHE_LINE_LEN - 1) / CACHE_LINE_LEN * CACHE_LINE_LEN;
double sum_times[MAX_NUM_THREADS][nlayers][NUM_BREAKDOWNS_ROUNDED_UP] = {0},
       sum_flops[nlayers][NUM_BREAKDOWNS] = {0};

class FullyConnectedForward {
 public:
  FullyConnectedForward(
      Matrix<float, PAD>* input,
      Matrix<float, PAD>* weight,
      Matrix<float, PAD>* output,
      int numa_node_id,
      int layer_id,
      int iteration)
      : input_(input),
        weight_(weight),
        output_(output),
        numa_node_id_(numa_node_id),
        layer_id_(layer_id),
        iteration_(iteration) {}

  void operator()() const {
    double t0 = dsecnd();
    int m = input_->nrows(), n = output_->ncols(), k = input_->ncols();

    // forward gemm performs well with aspect ratio
    // (m_end - m_begin) ~= 4 * (n_end - n_begin)
    float aspect_ratio = 4.;

    tbb::parallel_for(
        0,
        nthreads_per_socket,
        [&](size_t task_id) {
          int m_begin, m_end, n_begin, n_end;
          get_2dpartition(
              &m_begin,
              &m_end,
              &n_begin,
              &n_end,
              m,
              n,
              aspect_ratio,
              false /* m_align */,
              numa_node_id_,
              nthreads_per_socket,
              task_id);

          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasTrans,
              m_end - m_begin,
              n_end - n_begin,
              k,
              1.0f,
              input_->rawData(m_begin, 0),
              input_->ld(),
              weight_->rawData(numa_node_id_ * n + n_begin, 0),
              weight_->ld(),
              0.0f,
              output_->rawData(m_begin, n_begin),
              output_->ld());

          if (iteration_ >= NWARMUP) {
            int tid = numa_node_id_ * nthreads_per_socket +
                tbb::task_arena::current_thread_index();

            double dt = dsecnd() - t0;
            double flops = 2. * m * n * k;
#ifdef PRINT_PER_LAYER_PERFORMANCE
            // if (tid == 0) {
            double gflops = flops / dt / 1e9;
            cerr << "fwd layer " << layer_id_ << " tid " << tid << " tid "
                 << this_thread::get_id() << " " << m_end - m_begin << " x "
                 << n_end - n_begin << " x " << k << " " << dt * 1e3 << " ms "
                 << gflops << " GF/s " << nthreads << " GF/s/core" << endl;
        // }
#endif
            sum_times[tid][layer_id_][FWD] += dt;
            if (tid == 0) {
              sum_flops[layer_id_][FWD] += flops;
            }
          }
        },
        tbb::simple_partitioner());

    ++iteration_;
  }

  void operator()(flow::continue_msg) {
    FullyConnectedForward::operator()();
  }

 private:
  Matrix<float, PAD>*input_, *weight_, *output_;
  int numa_node_id_, layer_id_;
  mutable int iteration_;
};

class FullyConnectedBackward {
 public:
  FullyConnectedBackward(
      Matrix<float, PAD>* input,
      Matrix<float, PAD>* output_grad,
      Matrix<float, PAD>* weight,
      Matrix<float, PAD>* input_grad,
      Matrix<float, PAD>* weight_grad,
      int numa_node_id,
      int layer_id,
      int iteration)
      : input_(input),
        output_grad_(output_grad),
        weight_(weight),
        input_grad_(input_grad),
        weight_grad_(weight_grad),
        numa_node_id_(numa_node_id),
        layer_id_(layer_id),
        iteration_(iteration) {}

  void operator()() const {
    // weight gradient computation
    double t0 = dsecnd();

    int m = output_grad_->ncols(), n = input_->ncols(), k = input_->nrows();

    tbb::parallel_for(
        0,
        nthreads_per_socket,
        [&](size_t task_id) {
          int tid = numa_node_id_ * nthreads_per_socket +
              tbb::task_arena::current_thread_index();
          int m_begin, m_end, n_begin, n_end;

          // partition k over socket
          int k_per_socket = (k + nsockets - 1) / nsockets;
          int k_begin = std::min(numa_node_id_ * k_per_socket, k);
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
              nthreads_per_socket,
              task_id);
          if (0 == iteration_ && 0 == tid) {
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
            printf("numa_node = %d\n", numa_node_of_cpu(sched_getcpu()));
          }

          const float* A_begin = output_grad_->rawData(k_begin, m_begin);
          cblas_sgemm(
              CblasRowMajor,
              CblasTrans,
              CblasNoTrans,
              m_end - m_begin,
              n_end - n_begin,
              k_end - k_begin,
              1.0f,
              A_begin,
              output_grad_->ld(),
              input_->rawData(k_begin, n_begin),
              input_->ld(),
              0.0f,
              weight_grad_->rawData(numa_node_id_ * m + m_begin, n_begin),
              weight_grad_->ld());

          if (iteration_ >= NWARMUP) {
            double dt = dsecnd() - t0;
            double flops = 2. * m * n * k;
#ifdef PRINT_PER_LAYER_PERFORMANCE
            if (tid == 0) {
              double gflops = flops / dt / 1e9;
              printf(
                  "wgt_gradient layer %d %g ms %g GF/s %g GF/s/core\n",
                  layer_id_,
                  dt * 1e3,
                  gflops,
                  gflops / nthreads);
            }
#endif
            sum_times[tid][layer_id_][WGT_GRAD] += dt;
            if (tid == 0) {
              sum_flops[layer_id_][WGT_GRAD] += flops;
            }
          }
        },
        tbb::simple_partitioner());

    // backward update
    t0 = dsecnd();

    m = input_->nrows(), n = input_->ncols(), k = output_grad_->ncols();
    tbb::parallel_for(
        0,
        nthreads_per_socket,
        [&](size_t task_id) {
          int tid = numa_node_id_ * nthreads_per_socket +
              tbb::task_arena::current_thread_index();
          int m_begin, m_end, n_begin, n_end;
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
              numa_node_id_,
              nthreads_per_socket,
              task_id);
          if (0 == iteration_ && 0 == tid) {
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
            printf("numa_node = %d\n", numa_node_of_cpu(sched_getcpu()));
          }

          const float* A_begin = output_grad_->rawData(m_begin, 0);
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              m_end - m_begin,
              n_end - n_begin,
              k,
              1.0f,
              A_begin,
              output_grad_->ld(),
              weight_->rawData(numa_node_id_ * k, n_begin),
              weight_->ld(),
              0.0f,
              input_grad_->rawData(m_begin, n_begin),
              input_grad_->ld());

          if (iteration_ >= NWARMUP) {
            double dt = dsecnd() - t0;
            double flops = 2. * m * n * k;
#ifdef PRINT_PER_LAYER_PERFORMANCE
            if (tid == 0) {
              double gflops = flops / dt / 1e9;
              printf(
                  "bwd layer %d %g ms %g GF/s %g GF/s/core\n",
                  layer_id_,
                  dt * 1e3,
                  gflops,
                  gflops / nthreads);
            }
#endif
            sum_times[tid][layer_id_][BWD] += dt;
            if (tid == 0) {
              sum_flops[layer_id_][BWD] += flops;
            }
          }
        },
        tbb::simple_partitioner());

    ++iteration_;
  }

  void operator()(flow::continue_msg) {
    FullyConnectedBackward::operator()();
  }

 private:
  Matrix<float, PAD>*input_, *output_grad_, *weight_, *input_grad_,
      *weight_grad_;
  int numa_node_id_, layer_id_;
  mutable int iteration_;
};

// Create a thin async_node at each cross-graph edge.
// It prevents task bypassing which would violate affinity.
// The passed-in graph object must be the receiver's graph
// (unfortunately for now it cannot be obtained from the receiver).
// Returns: unique_ptr that holds the created async_node.
template <typename T>
std::unique_ptr<flow::graph_node> make_crossgraph_edge(
    flow::sender<T>& s,
    flow::receiver<T>& r,
    flow::graph& receiver_g) {
  typedef flow::async_node<T, T> async_node_t;
  auto a = new async_node_t(
      receiver_g,
      flow::unlimited,
      [](T msg, typename async_node_t::gateway_type& gw) { gw.try_put(msg); });
  flow::make_edge(s, *a);
  flow::make_edge(*a, r);
  return std::unique_ptr<flow::graph_node>(a);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "%s <nsockets> <nthreads_per_socket>\n", argv[0]);
    exit(1);
  }

  nsockets = atoi(argv[1]);
  nthreads_per_socket = atoi(argv[2]);
  nthreads = nsockets * nthreads_per_socket;
  omp_set_num_threads(1);

  tbb::task_scheduler_init scheduler_init(nthreads);
  for (int s = 0; s < nsockets; ++s) {
    tbb_arena.emplace_back(new tbb::task_arena(nthreads_per_socket, s == 0));
    tbb_observers.emplace_back(new pinning_observer(*tbb_arena[s], s));
  }

  int batch_size = 1024 * nsockets; // weak-scaling with nsockets

  unique_ptr<Matrix<float, PAD>> weights[nlayers], weight_grads[nlayers],
      activations[nlayers + 2];

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

  vector<tbb::task_group> tg(nsockets);

  vector<flow::graph> dags(nsockets);
  for (int sid = 0; sid < nsockets; ++sid) {
    tbb_arena[sid]->execute([&dags, sid] { dags[sid].reset(); });
  }

  using namespace tbb::flow;

  continue_node<continue_msg> dag_root(
      dags[0], [&dags](continue_msg) { dags[0].reserve_wait(); });
  continue_node<continue_msg> dag_exit(
      dags[0], [&dags](continue_msg) { dags[0].release_wait(); });

  vector<unique_ptr<continue_node<continue_msg, lightweight>>> tbb_flow_nodes;
  vector<unique_ptr<graph_node>> cross_graph_edges;
  // forward
  for (int l = 0; l < nlayers; ++l) {
    for (int numa_node_id = 0; numa_node_id < nsockets; ++numa_node_id) {
      tbb_flow_nodes.emplace_back(new continue_node<continue_msg, lightweight>(
          dags[numa_node_id],
          FullyConnectedForward(
              activations[l].get(),
              weights[l].get(),
              activations[l + 1].get(),
              numa_node_id,
              l,
              0)));
      if (l == 0) {
        if (numa_node_id == 0) {
          make_edge(dag_root, *tbb_flow_nodes.back());
        } else {
          cross_graph_edges.push_back(make_crossgraph_edge(
              dag_root, *tbb_flow_nodes.back(), dags[numa_node_id]));
        }
      } else {
        make_edge(
            *tbb_flow_nodes[(l - 1) * nsockets + numa_node_id],
            *tbb_flow_nodes.back());
      }
    } // for each socket
  } // for each layer
  // backward
  for (int l = nlayers - 1; l >= 0; --l) {
    for (int numa_node_id = 0; numa_node_id < nsockets; ++numa_node_id) {
      tbb_flow_nodes.emplace_back(new continue_node<continue_msg, lightweight>(
          dags[numa_node_id],
          FullyConnectedBackward(
              activations[l].get(), // input
              activations[l + 1].get(), // output_grad
              weights[l].get(),
              activations[l == 0 ? nlayers + 1 : l].get(), // input_grad
              weight_grads[l].get(),
              numa_node_id,
              l,
              0)));

      make_edge(
          *tbb_flow_nodes
              [(nlayers + (nlayers - 1 - l) - 1) * nsockets + numa_node_id],
          *tbb_flow_nodes.back());

      if (l == 0) {
        if (numa_node_id == 0) {
          make_edge(*tbb_flow_nodes.back(), dag_exit);
        } else {
          cross_graph_edges.push_back(
              make_crossgraph_edge(*tbb_flow_nodes.back(), dag_exit, dags[0]));
        }
      }
    } // for each socket
  } // for each layer

  bool use_flow_graph = true;
  for (int it = 0; it < NWARMUP + NITER; ++it) {
    if (use_flow_graph) {
      tbb_arena[0]->execute([&dag_root] { dag_root.try_put(continue_msg()); });
      tbb_arena[0]->execute([&dags] { dags[0].wait_for_all(); });
      continue;
    }

    // forward
    for (int l = 0; l < nlayers; ++l) {
      for (int sid = nsockets - 1; sid >= 1; --sid) {
        tbb_arena[sid]->enqueue([&, sid, l, it] {
          tg[sid].run(FullyConnectedForward(
              activations[l].get(),
              weights[l].get(),
              activations[l + 1].get(),
              sid,
              l,
              it));
        });
      } // sid

      int sid = 0;
      tbb_arena[sid]->execute([&, sid, l, it] {
        tg[sid].run(FullyConnectedForward(
            activations[l].get(),
            weights[l].get(),
            activations[l + 1].get(),
            sid,
            l,
            it));
      });

      tbb_arena[0]->execute([&tg] { tg[0].wait(); });
    } // for each layer

    // backward
    for (int l = nlayers - 1; l >= 0; --l) {
      for (int sid = nsockets - 1; sid >= 1; --sid) {
        tbb_arena[sid]->enqueue([&, sid, l, it] {
          tg[sid].run(FullyConnectedBackward(
              activations[l].get(), // input
              activations[l + 1].get(), // output_grad
              weights[l].get(),
              activations[l == 0 ? nlayers + 1 : l].get(), // input_grad
              weight_grads[l].get(),
              sid,
              l,
              it));
        });
      } // sid

      int sid = 0;
      tbb_arena[sid]->execute([&, sid, l, it] {
        tg[sid].run(FullyConnectedBackward(
            activations[l].get(), // input
            activations[l + 1].get(), // output_grad
            weights[l].get(),
            activations[l == 0 ? nlayers + 1 : l].get(), // input_grad
            weight_grads[l].get(),
            sid,
            l,
            it));
      });

      tbb_arena[0]->execute([&tg] { tg[0].wait(); });
    } // for each layer
  } // for each iteration

  /////////////////////////////////////////////////////////////////////////////
  // compute load imbalance
  double load_imbalance[nlayers][NUM_BREAKDOWNS];
  double max_sum_times[nlayers][NUM_BREAKDOWNS];

  for (int l = 0; l < nlayers; ++l) {
    for (int i = FWD; i < NUM_BREAKDOWNS; ++i) {
      double sum = 0, max = 0;
      if (i == WGT_GRAD || i == BWD) {
        for (int sid = 0; sid < nsockets; ++sid) {
          for (int tid_in_socket = 0; tid_in_socket < nthreads_per_socket;
               ++tid_in_socket) {
            int tid = sid * nthreads_per_socket + tid_in_socket;
            sum += sum_times[tid][l][i];
            max = std::max(max, sum_times[tid][l][i]);
          }
        }
        max_sum_times[l][i] = max;

        double avg = sum / nthreads_per_socket / nsockets;
        load_imbalance[l][i] = max / avg;
      } else {
        // i >= WGT_UPDATE
        // ? nthreads_per_socket_for_allreduce[nthreads_per_socket] * nsockets
        // : nthreads;

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
  double total_times[NUM_BREAKDOWNS] = {0}, total_flops[NUM_BREAKDOWNS] = {0};
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
