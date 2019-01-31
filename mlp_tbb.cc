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
#include <x86intrin.h>
#include <sstream>
#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include <tbb/task_scheduler_observer.h>
#include <tbb/task_scheduler_init.h>

#include "Matrix.h"
#include "Partition.h"
#include "Rand.h"
#include <pthread.h>
#include <iostream>
#if COUNT_NODES
#include <tbb/atomic.h>
tbb::atomic<int> num_node_executions;
#endif
// #define PRINT_PER_LAYER_PERFORMANCE

#if POST_VALIDATION
#include <mutex>
std::mutex mtx;
#endif
using namespace std;
namespace flow = tbb::flow;

constexpr int PAD = 16;
constexpr int CACHE_LINE_LEN = 16;
int nthreads_per_socket;

thread_local int my_last_sid = -1;

class pinning_observer : public tbb::task_scheduler_observer {
 public:
  pinning_observer(tbb::task_arena& arena, int numa_node_id)
      : tbb::task_scheduler_observer(arena), arena_(arena), numa_node_id_(numa_node_id) {
    observe(true);
  } // activate the observer

  void on_scheduler_entry(bool /* unused */) override {
#if THREAD_INFO
        std::stringstream ss;
        ss << "OBSERVER ENTRY," << std::this_thread::get_id() << "," << numa_node_id_ <<","  << this  << std::endl;
        printf("%s", ss.str().c_str());
#endif
        if(my_last_sid == numa_node_id_){
            return;
        }
        my_last_sid = numa_node_id_;
#if NUMA_BIND
        auto bm = numa_allocate_nodemask();
        numa_bitmask_clearall(bm);
        numa_bitmask_setbit(bm, numa_node_id_);
        numa_bind(bm);
        numa_bitmask_free(bm);
#endif
#if CORE_PINNING
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
            for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
                CPU_SET(i, &cpuset);
            }
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
       }
#endif
  }

  void on_scheduler_exit(bool /*unused*/){
#if THREAD_INFO
    std::stringstream ss;
    ss << "OBSERVER EXIT," << std::this_thread::get_id() << "," << numa_node_id_ <<","  << this  << std::endl;
    printf("%s", ss.str().c_str());
#endif
  }

private:
  tbb::task_arena& arena_;
  int numa_node_id_;
};

vector<unique_ptr<tbb::task_arena>> tbb_arena;
vector<unique_ptr<pinning_observer>> tbb_observers;
vector<unique_ptr<tbb::affinity_partitioner>> tbb_affinity_partitioners;

Matrix<float, PAD> *
  create_matrix_with_numa_aware_allocation(int nrows, int ncols)
{
  Matrix<float, PAD> *matrix = new Matrix<float, PAD>(nrows, ncols);

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
int nfeatures[] = { 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1 };
//int nfeatures[] = { 512, 512 };
constexpr int nlayers = sizeof(nfeatures)/sizeof(nfeatures[0]) - 1;
constexpr int MAX_NUM_THREADS = 1024;
constexpr int NWARMUP = 2, NITER = 256;

// Being careful not to have false sharing
constexpr int NUM_BREAKDOWNS_ROUNDED_UP =
  (NUM_BREAKDOWNS + CACHE_LINE_LEN - 1) / CACHE_LINE_LEN * CACHE_LINE_LEN;
double
  sum_times[MAX_NUM_THREADS][nlayers][NUM_BREAKDOWNS_ROUNDED_UP] = { 0 },
  sum_flops[nlayers][NUM_BREAKDOWNS] = { 0 };

class FullyConnectedForward {
 public:
  FullyConnectedForward(
      Matrix<float, PAD>* input,
      Matrix<float, PAD>* weight,
      Matrix<float, PAD>* output,
      int numa_node_id,
      int layer_id,
      int iteration,
      vector<unique_ptr<tbb::affinity_partitioner>> *tbb_affinity_partitioners)
      : input_(input),
        weight_(weight),
        output_(output),
        numa_node_id_(numa_node_id),
        layer_id_(layer_id),
        tbb_affinity_partitioners_(tbb_affinity_partitioners),
        iteration_(iteration) {}

  void operator()() const {
    double t0 = dsecnd();
    int m = input_->nrows(), n = output_->ncols(), k = input_->ncols();

#if COUNT_NODES
    ++num_node_executions;
#endif
    // forward gemm performs well with aspect ratio
    // (m_end - m_begin) ~= 4 * (n_end - n_begin)
    float aspect_ratio = 4.;

    tbb::parallel_for(
      0,
      SP*nthreads_per_socket,
      [&](size_t task_id) {
        double sgst = dsecnd();
        int tid = numa_node_id_ * nthreads_per_socket + task_id;
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
            SP*nthreads_per_socket,
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
          double dt = dsecnd() - t0;
          double flops = 2. * m * n * k;
#ifdef PRINT_PER_LAYER_PERFORMANCE
          // if (tid == 0) {
            double gflops = flops / dt / 1e9;
            // LOG(INFO) << "fwd layer " << l << " tid " << tid << " tid "
            //      << this_thread::get_id() << " " << m_end - m_begin
            //      << " x " << n_end - n_begin << " x " << k << " "
            //      << dt * 1e3 << " ms " << gflops << " GF/s "
            //      << gflops / nthreads << " GF/s/core";
          // }
#endif
          sum_times[tid][layer_id_][FWD] += dt;
          if (tid == 0) {
            sum_flops[layer_id_][FWD] += flops;
          }
        }
        double sget = dsecnd() - sgst;
#if THREAD_INFO
        std::stringstream ss;
        ss << "SGEMM," << iteration_ << "," << numa_node_id_ <<","  << layer_id_ <<"," << task_id << "," << std::this_thread::get_id() << "," << sget << std::endl;
        printf("%s", ss.str().c_str());
#endif

      },
      tbb::simple_partitioner());

#if POST_VALIDATION
      mtx.lock();
      if (iteration_ == 257 && layer_id_ == 6)
      {
        float* fptr = output_->rawData(); 
        for(int i = 0; i<100 ;i++)
        {
           if(i==0)
              cout << "Validation - Socket ID:" << numa_node_id_ << std::endl; 
           cout << *fptr << ",";
           fptr++;
        }
        cout << std::endl;
      }
      mtx.unlock();

#endif
    ++iteration_;
  }

  void operator()(flow::continue_msg) {
    FullyConnectedForward::operator()();
  }

 private:
  Matrix<float, PAD> *input_, *weight_, *output_;
  int numa_node_id_, layer_id_;
  mutable int iteration_;
  vector<unique_ptr<tbb::affinity_partitioner>> *tbb_affinity_partitioners_;
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

int main(int argc, char **argv)
{
  if (argc != 4) {
    fprintf(stderr, "%s <batch_size> <nsockets/no+of_arenas> <nthreads_per_socket_or_arenas>\n", argv[0]);
    exit(1);
  }
#if THREAD_INFO
  std::stringstream ss;
  ss << "MAIN THREAD," << std::this_thread::get_id() << std::endl;
  printf("%s", ss.str().c_str());
#endif
#if CORE_PINNING
  printf("Enabled Core Pinning\n");
#endif
#if NUMA_BIND
  printf("Enabled NUMA Bind\n");
#endif
#if COUNT_NODES
  num_node_executions = 0;
#endif
  nsockets = atoi(argv[2]);
  nthreads_per_socket = atoi(argv[3]);
  int batch_size = atoi(argv[1]); // weak-scaling with nsockets

  int nthreads = nsockets * nthreads_per_socket;
  // omp_set_num_threads(1);
  tbb::task_scheduler_init scheduler_init(nthreads);
  // tbb_affinity_partitioners.resize(nsockets);
  for (int s = 0; s < nsockets; ++s) {
    tbb_arena.emplace_back(new tbb::task_arena(nthreads_per_socket, s == 0));
    tbb_observers.emplace_back(new pinning_observer(*tbb_arena[s], s));
  }

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

  vector<tbb::task_group> tg(nsockets);

  vector<flow::graph> dags(nsockets);
  for (int sid = 0; sid < nsockets; ++sid) {
    tbb_arena[sid]->execute([&dags, sid] { dags[sid].reset(); });
  }

  using namespace tbb::flow;

#if USE_BROADCAST_NODE
  broadcast_node<continue_msg> dag_root(dags[0]);
#else
  continue_node<continue_msg> dag_root(dags[0], [&dags](continue_msg) {
    dags[0].reserve_wait();
  });
#endif
  continue_node<continue_msg> dag_exit(dags[0], [&dags](continue_msg) {
    dags[0].release_wait();
  });

#if USE_LIGHTWEIGHT
  printf("Using Lightweight except first layer\n");
  typedef continue_node<continue_msg> cn_type;
  typedef continue_node<continue_msg, lightweight> light_cn_type;
  vector<unique_ptr<light_cn_type>> light_tbb_flow_nodes;
  vector<unique_ptr<cn_type>> tbb_flow_nodes;
  vector<unique_ptr<graph_node>> cross_graph_edges;
  for (int l = 0; l < nlayers; ++l) {
    for (int numa_node_id = 0; numa_node_id < nsockets; ++numa_node_id) {
        if( l == 0) {
            tbb_flow_nodes.emplace_back(
                new cn_type(
                    dags[numa_node_id],
                    FullyConnectedForward(
                        activations[l].get(), weights[l].get(), activations[l + 1].get(),
                        numa_node_id, l, 0, &tbb_affinity_partitioners)
                )
            );

            if (numa_node_id == 0) {
                make_edge(dag_root, *tbb_flow_nodes.back());
            }
            else {
                cross_graph_edges.push_back(
                make_crossgraph_edge(
                    dag_root, *tbb_flow_nodes.back(), dags[numa_node_id]));
            }
        } else {
            light_tbb_flow_nodes.emplace_back(
                new light_cn_type(
                    dags[numa_node_id],
                    FullyConnectedForward(
                        activations[l].get(), weights[l].get(), activations[l + 1].get(),
                        numa_node_id, l, 0, &tbb_affinity_partitioners)
               )
            );
            if(l==1){
                make_edge(
                    *tbb_flow_nodes[(l - 1) * nsockets + numa_node_id],
                    *light_tbb_flow_nodes.back());
            }
            else {
                make_edge(
                    *light_tbb_flow_nodes[(l - 2) * nsockets + numa_node_id],
                    *light_tbb_flow_nodes.back());
            }
        }

        if (l == nlayers - 1) {
            if (numa_node_id == 0) {
                make_edge(*light_tbb_flow_nodes.back(), dag_exit);
            } else {
                cross_graph_edges.push_back(
                make_crossgraph_edge(*light_tbb_flow_nodes.back(), dag_exit, dags[0]));
            }
        }
    } // for each socket
  } // for each layer
#else 
  typedef continue_node<continue_msg> cn_type;
  vector<unique_ptr<cn_type>> tbb_flow_nodes;
  vector<unique_ptr<graph_node>> cross_graph_edges;
  for (int l = 0; l < nlayers; ++l) {
    for (int numa_node_id = 0; numa_node_id < nsockets; ++numa_node_id) {
      tbb_flow_nodes.emplace_back(
        new cn_type(
            dags[numa_node_id],
            FullyConnectedForward(
               activations[l].get(), weights[l].get(), activations[l + 1].get(),
               numa_node_id, l, 0, &tbb_affinity_partitioners)
        ));
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

      if (l == nlayers - 1) {
        if (numa_node_id == 0) {
          make_edge(*tbb_flow_nodes.back(), dag_exit);
        } else {
          cross_graph_edges.push_back(
              make_crossgraph_edge(*tbb_flow_nodes.back(), dag_exit, dags[0]));
        }
      }
    } // for each socket
  } // for each layer
#endif

//  bool use_flow_graph = false;
//  for (int it = 0; it < NWARMUP + NITER; ++it) {
//    if (use_flow_graph) {
//      tbb_arena[0]->execute([&dag_root] { dag_root.try_put(continue_msg()); });
//      tbb_arena[0]->execute([&dags] { dags[0].wait_for_all(); });
//      continue;
//    }

#if USE_FG 
     printf("Using Flow Graph\n");
#endif
#if TIME_FG_LOOP
    double fg_t0 = 0;
#endif
  for (int it = 0; it < NWARMUP + NITER; ++it) {
#if THREAD_INFO
        printf("ITERATION,%d\n", it);
#endif
#if TIME_FG_LOOP
      if (it == NWARMUP) 
        fg_t0 = dsecnd();
#endif
#if USE_FG
#if USE_BROADCAST_NODE
        dags[0].reserve_wait();
        dag_root.try_put(continue_msg());
        tbb_arena[0]->execute(
            [&dags] {
                dags[0].wait_for_all();
            }
        );
#else
        tbb_arena[0]->execute(
            [&dag_root, &dags] {
                dag_root.try_put(continue_msg());
                dags[0].wait_for_all();
            }
        );
#endif
        continue;
#endif

    // forward
    for (int l = 0; l < nlayers; ++l) {
      double t0 = dsecnd();

      int m = batch_size, n = nfeatures[l + 1], k = nfeatures[l];

      // forward gemm performs well with aspect ratio
      // (m_end - m_begin) ~= 32 * (n_end - n_begin)
      float aspect_ratio = 4.;

      for (int sid = nsockets - 1; sid >= 1; --sid) {
        tbb_arena[sid]->execute([&, sid, l, it]{ tg[sid].run(
          FullyConnectedForward(
            activations[l].get(), weights[l].get(), activations[l + 1].get(),
            sid, l, it, &tbb_affinity_partitioners)); });
      } // sid

      int sid = 0;
      tbb_arena[sid]->execute([&, sid, l, it]{ tg[sid].run(
        FullyConnectedForward(
          activations[l].get(), weights[l].get(), activations[l + 1].get(),
          sid, l, it, &tbb_affinity_partitioners)); });

      tbb_arena[0]->execute([&tg]{ tg[0].wait(); });
    } // for each layer

    // backward
//     for (int l = nlayers - 1; l >= 0; --l) {
//       // weight gradient computation
//       double t0 = dsecnd();
//
//       int m = nfeatures[l + 1], n = nfeatures[l], k = batch_size;
//
//       for (int sid = 0; sid < nsockets; ++sid) {
//         tbb_arena[sid]->execute([&]() {
//           tbb::parallel_for(
//               0,
//               nthreads_per_socket,
//               [&](size_t task_id) {
//                 int tid = sid * nthreads_per_socket + task_id;
//                 int m_begin, m_end, n_begin, n_end;
//
//                 // partition k over socket
//                 int k_per_socket = (k + nsockets - 1) / nsockets;
//                 int k_begin = std::min(sid * k_per_socket, k);
//                 int k_end = std::min(k_begin + k_per_socket, k);
//
//                 // 2d partition m and n within socket
//                 // weight_grad gemm performs well with aspect ratio
//                 // 8 * (m_end - m_begin) ~= (n_end - n_begin)
//                 float aspect_ratio = 1./2;
//                 /*if (m == 512 && n == 1280) {
//                   aspect_ratio = 2.;
//                 }*/
//                 get_intra_socket_2dpartition(
//                     &m_begin,
//                     &m_end,
//                     &n_begin,
//                     &n_end,
//                     m,
//                     n,
//                     aspect_ratio,
//                     true /* m_align */,
//                     nthreads_per_socket,
//                     task_id);
//                 if (0 == it && 0 == tid) {
//                   int mb = (m + m_end - m_begin - 1) / (m_end - m_begin);
//                   int nb = (n + n_end - n_begin - 1) / (n_end - n_begin);
//                   int kb = k / nsockets / (k_end - k_begin);
//                   printf(
//                       "wgt m %d n %d k %d bm %d bn %d bk %d mb %d nb %d kb %d\n",
//                       m,
//                       n,
//                       k,
//                       m_end - m_begin,
//                       n_end - n_begin,
//                       k_end - k_begin,
//                       mb,
//                       nb,
//                       kb);
//                   printf("numa_node = %d\n", numa_node_of_cpu(sched_getcpu()));
//                 }
//
//                 const float* A_begin =
//                     activations[l + 1]->rawData(k_begin, m_begin);
//                 cblas_sgemm(
//                     CblasRowMajor,
//                     CblasTrans,
//                     CblasNoTrans,
//                     m_end - m_begin,
//                     n_end - n_begin,
//                     k_end - k_begin,
//                     1.0f,
//                     A_begin,
//                     activations[l + 1]->ld(),
//                     activations[l]->rawData(k_begin, n_begin),
//                     activations[l]->ld(),
//                     0.0f,
//                     weight_grads[l]->rawData(sid * m + m_begin, n_begin),
//                     weight_grads[l]->ld());
//
//                 if (it >= NWARMUP) {
//                   double dt = dsecnd() - t0;
//                   double flops = 2. * m * n * k;
// #ifdef PRINT_PER_LAYER_PERFORMANCE
//                   if (tid == 0) {
//                     double gflops = flops / dt / 1e9;
//                     printf(
//                       "wgt_gradient layer %d %g ms %g GF/s %g GF/s/core\n",
//                       l, dt * 1e3, gflops, gflops / nthreads);
//                   }
// #endif
//                   sum_times[tid][l][WGT_GRAD] += dt;
//                   if (tid == 0) {
//                     sum_flops[l][WGT_GRAD] += flops;
//                   }
//                 }
//               },
//               *tbb_affinity_partitioners[sid]);
//         }); // arena execute
//       } // sid
//
//       // backward update
//       t0 = dsecnd();
//
//       m = batch_size, n = nfeatures[l], k = nfeatures[l + 1];
//       for (int sid = 0; sid < nsockets; ++sid) {
//         tbb_arena[sid]->execute([&]() {
//           tbb::parallel_for(
//               0,
//               nthreads_per_socket,
//               [&](size_t task_id) {
//                 int tid = sid * nthreads_per_socket + task_id;
//                 int m_begin, m_end, n_begin, n_end;
//                 // backward gemm performs well with aspect ratio
//                 // (m_end - m_begin) ~= 32 * (n_end - n_begin)
//                 float aspect_ratio = 1./2;
//                 /*if (n == 512 && k == 512) {
//                   aspect_ratio = 4.;
//                 }
//                 else if (n == 1280 && k == 512) {
//                   aspect_ratio = 16.;
//                 }*/
//                 get_2dpartition(
//                     &m_begin,
//                     &m_end,
//                     &n_begin,
//                     &n_end,
//                     m,
//                     n,
//                     aspect_ratio,
//                     false /* m_align */,
//                     sid,
//                     nthreads_per_socket,
//                     task_id);
//                 if (0 == it && 0 == tid) {
//                   int mb =
//                       (m / nsockets + m_end - m_begin - 1) / (m_end - m_begin);
//                   int nb = (n + n_end - n_begin - 1) / (n_end - n_begin);
//                   printf(
//                       "bwd m %d n %d k %d bm %d bn %d bk %d mb %d nb %d kb %d\n",
//                       m,
//                       n,
//                       k,
//                       m_end - m_begin,
//                       n_end - n_begin,
//                       k,
//                       mb,
//                       nb,
//                       1);
//                   printf("numa_node = %d\n", numa_node_of_cpu(sched_getcpu()));
//                 }
//
//                 const float *A_begin = activations[l + 1]->rawData(m_begin, 0);
//                 Matrix<float, PAD>* C =
//                     activations[l == 0 ? nlayers + 1 : l].get();
//                 cblas_sgemm(
//                     CblasRowMajor,
//                     CblasNoTrans,
//                     CblasNoTrans,
//                     m_end - m_begin,
//                     n_end - n_begin,
//                     k,
//                     1.0f,
//                     A_begin,
//                     activations[l + 1]->ld(),
//                     weights[l]->rawData(sid * k, n_begin),
//                     weights[l]->ld(),
//                     0.0f,
//                     C->rawData(m_begin, n_begin),
//                     C->ld());
//
//                 if (it >= NWARMUP) {
//                   double dt = dsecnd() - t0;
//                   double flops = 2. * m * n * k;
// #ifdef PRINT_PER_LAYER_PERFORMANCE
//                   if (tid == 0) {
//                     double gflops = flops / dt / 1e9;
//                     printf(
//                       "bwd layer %d %g ms %g GF/s %g GF/s/core\n",
//                       l, dt * 1e3, gflops, gflops / nthreads);
//                   }
// #endif
//                   sum_times[tid][l][BWD] += dt;
//                   if (tid == 0) {
//                     sum_flops[l][BWD] += flops;
//                   }
//                 }
//               },
//               *tbb_affinity_partitioners[sid]);
//         }); // arena execute
//       } // sid
//     } // for each layer
  } // for each iteration
#if TIME_FG_LOOP
  double fg_total_time = dsecnd() - fg_t0;
#endif

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
        int nthreads_for_i = nthreads;
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
  double
    total_times[NUM_BREAKDOWNS] = { 0 }, total_flops[NUM_BREAKDOWNS] = { 0 };
  for (int l = 0; l < nlayers; ++l) {
#if POST_VALIDATION
#else
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
#endif
    for (int i = FWD; i < NUM_BREAKDOWNS; ++i) {
      total_times[i] += max_sum_times[l][i];
      total_flops[i] += sum_flops[l][i];
    }
  } // for each layer

#if POST_VALIDATION
#else
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
#endif
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
#if POST_VALIDATION
#else
    printf("layer %d l1 %g l2 %g trace %g\n", l, l1_norm, l2_norm, trace);
#endif
  }

#if COUNT_NODES
  int nne = num_node_executions;
  printf("number of node executions = %d\n", nne);
#endif
#if TIME_FG_LOOP
  printf("total time for fg loop = %g\n", fg_total_time);
#endif
  return 0;
}
