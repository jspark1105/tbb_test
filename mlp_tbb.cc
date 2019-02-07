#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>

#include <mkl.h>
#include <numa.h>
#include <omp.h>
#include <immintrin.h>
// #include <glog/logging.h>

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
#include "mlp_bench_common.h"

// #define PRINT_PER_LAYER_PERFORMANCE
// #define CORE_AFFINITY
// #define NO_ALL_REDUCE
#define COLLECT_TRACE

// call numa bind again at the beginning of FCGradient
#define SECOND_NUMA_BIND
#define USE_LIGHTWEIGHT_IN_ALL_REDUCE

using namespace std;
namespace flow = tbb::flow;

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
#ifdef CORE_AFFINITY
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
#endif
  }

 private:
  int numa_node_id_;
};

#ifdef COLLECT_TRACE
namespace {

struct Trace {
  long timestamp_us;
  int pid, tid;
  string name;
  bool begin;

  void serialize(ostream& ost) {
    ost << "{\n";
    ost << " \"ts\": " << timestamp_us << ",\n";
    ost << " \"pid\": " << pid << ",\n";
    ost << " \"tid\": " << tid << ",\n";
    if (begin) {
      ost << " \"name\": \"" << name << "\",\n";
      ost << " \"ph\": \"B\"\n";
    } else {
      ost << " \"ph\": \"E\"\n";
    }
    ost << "}";
  }
};
}

vector<vector<Trace>> traces;
#endif

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

// Reduce-scatter phase or AllReduce
// We have num_steps * ntasks_per_socket_for_allreduce tasks for each weight
// where num_steps = nsockts - 1
class UpdateWeightReduceScatter {
 public:
  UpdateWeightReduceScatter(Matrix<float, PAD>* weight_grad,
                            Matrix<float, PAD>* weight_grad_push_buf,
                            int layer_id,
                            int step,
                            int numa_node_id,
                            int task,
                            int idx_in_ring,
                            int next_sid)
      : weight_grad_(weight_grad),
        weight_grad_push_buf_(weight_grad_push_buf),
        layer_id_(layer_id),
        step_(step),
        numa_node_id_(numa_node_id),
        task_id_(task),
        idx_in_ring_(idx_in_ring),
        next_sid_(next_sid) {}

  void operator()() const {
    int tid = numa_node_id_ * nthreads_per_socket +
        tbb::task_arena::current_thread_index();

#ifdef COLLECT_TRACE
    long time_stamp = chrono::time_point_cast<chrono::microseconds>(
                          chrono::steady_clock::now())
                          .time_since_epoch()
                          .count();
    if (iteration_ == NWARMUP - 1) {
      traces[tid].emplace_back(Trace{time_stamp,
                                     numa_node_id_,
                                     tid - numa_node_id_ * nthreads_per_socket,
                                     string("ReduceScatter") + "_" +
                                         to_string(nlayers - 1 - layer_id_) +
                                         "_" + to_string(task_id_),
                                     true /* begin */});
    }
#endif

    int nrows = weight_grad_->nrows() / nsockets;
    int ncols = weight_grad_->ncols();

    int sid = numa_node_id_;
    double t_reduce_scatter_begin = dsecnd();

    int ld = weight_grad_->ld();
    size_t weight_size = nrows * ld;
    assert(weight_size % CACHE_LINE_LEN == 0);

    int ntasks_per_socket =
        nthreads_per_socket_for_allreduce[nthreads_per_socket];

    size_t i_per_chunk = (weight_size + nsockets * CACHE_LINE_LEN - 1) /
        nsockets / CACHE_LINE_LEN * CACHE_LINE_LEN;
    size_t i_per_task =
        (i_per_chunk + ntasks_per_socket * CACHE_LINE_LEN - 1) /
        ntasks_per_socket / CACHE_LINE_LEN * CACHE_LINE_LEN;

    size_t socket_begin = sid * weight_size;
    size_t next_socket_begin = next_sid_ * weight_size;

    // we partition the array into nsockets chunks
    // at ith step, socket s reads (nsockets - 1 + s - i)th chunk from
    // socket s - 1 and accumulates to its local chunk
    int chunk_to_push = (idx_in_ring_ - step_ + nsockets) % nsockets;
    int chunk_to_read = (chunk_to_push + nsockets) % nsockets;
    size_t chunk_begin = std::min(chunk_to_read * i_per_chunk, weight_size);
    size_t chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    size_t task_begin =
        std::min(chunk_begin + task_id_ * i_per_task, chunk_end);
    size_t task_end = std::min(task_begin + i_per_task, chunk_end);

    size_t dst_begin = socket_begin + task_begin;

    if (step_ > 0) {
      // accumulate wgt grads pushed from previous step
      for (size_t i = 0; i < task_end - task_begin; i += CACHE_LINE_LEN) {
        _mm512_stream_ps(
            weight_grad_->rawData() + dst_begin + i,
            _mm512_add_ps(
                _mm512_load_ps(weight_grad_->rawData() + dst_begin + i),
                _mm512_load_ps(
                    weight_grad_push_buf_->rawData() + dst_begin + i)));
      }
    }

    chunk_begin = std::min(chunk_to_push * i_per_chunk, weight_size);
    chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    task_begin = std::min(chunk_begin + task_id_ * i_per_task, chunk_end);
    task_end = std::min(task_begin + i_per_task, chunk_end);

    size_t src_begin = socket_begin + task_begin;
    dst_begin = next_socket_begin + task_begin;

    // push to buffer using non-temporal store
    for (size_t i = 0; i < task_end - task_begin; i += CACHE_LINE_LEN) {
      _mm512_stream_si512(
          reinterpret_cast<__m512i*>(
              weight_grad_push_buf_->rawData() + dst_begin + i),
          _mm512_load_si512(weight_grad_->rawData() + src_begin + i));
    }

    // Make sure non-temporal stores are fully visible to other threads
    _mm_sfence();

    if (iteration_ >= NWARMUP) {
      double dt = dsecnd() - t_reduce_scatter_begin;
      double bytes = (nsockets - 1) * nrows * ncols * sizeof(float);

      sum_times[tid][layer_id_][WGT_UPDATE_REDUCE_SCATTER] += dt;
      if (task_id_ == 0 && sid == 0 && step_ == 0) {
        sum_flops[layer_id_][WGT_UPDATE_REDUCE_SCATTER] += bytes;
      }
    }

#ifdef COLLECT_TRACE
    time_stamp = chrono::time_point_cast<chrono::microseconds>(
                          chrono::steady_clock::now())
                          .time_since_epoch()
                          .count();
    if (iteration_ == NWARMUP - 1) {
      traces[tid].emplace_back(Trace{time_stamp,
                                     numa_node_id_,
                                     tid - numa_node_id_ * nthreads_per_socket,
                                     string("ReduceScatter") + "_" +
                                         to_string(nlayers - 1 - layer_id_) +
                                         "_" + to_string(task_id_),
                                     false /* end */});
    }
#endif

    ++iteration_;
  }

  void operator()(flow::continue_msg) {
    UpdateWeightReduceScatter::operator()();
  }

 private:
  Matrix<float, PAD>* weight_grad_;
  Matrix<float, PAD>* weight_grad_push_buf_;
  int layer_id_, step_, numa_node_id_;
  int task_id_; // task id within socket
  int idx_in_ring_; // my id in ring
  int next_sid_; // next socket in the ring to send data
  mutable int iteration_{0};
};

class UpdateWeightAllGather {
 public:
  UpdateWeightAllGather(
      Matrix<float, PAD>* weight,
      Matrix<float, PAD>* weight_grad,
      Matrix<float, PAD>* weight_grad_push_buf,
      double alpha,
      int layer_id,
      int step,
      int numa_node_id,
      int task,
      int idx_in_ring,
      int next_sid)
      : weight_(weight),
        weight_grad_(weight_grad),
        weight_grad_push_buf_(weight_grad_push_buf),
        alpha_(alpha),
        layer_id_(layer_id),
        step_(step),
        numa_node_id_(numa_node_id),
        task_id_(task),
        idx_in_ring_(idx_in_ring),
        next_sid_(next_sid) {}

  void operator()() const {
    int tid = numa_node_id_ * nthreads_per_socket +
        tbb::task_arena::current_thread_index();

#ifdef COLLECT_TRACE
    long time_stamp = chrono::time_point_cast<chrono::microseconds>(
                          chrono::steady_clock::now())
                          .time_since_epoch()
                          .count();
    if (iteration_ == NWARMUP - 1) {
      traces[tid].emplace_back(Trace{time_stamp,
                                     numa_node_id_,
                                     tid - numa_node_id_ * nthreads_per_socket,
                                     string("AllGather") + "_" +
                                         to_string(nlayers - 1 - layer_id_) +
                                         "_" + to_string(task_id_),
                                     true /* begin */});
    }
#endif

    int nrows = weight_grad_->nrows() / nsockets;
    int ncols = weight_grad_->ncols();

    int sid = numa_node_id_;
    double t_allgather_begin = dsecnd();

    int ld = weight_grad_->ld();
    size_t weight_size = nrows * ld;
    assert(weight_size % CACHE_LINE_LEN == 0);

    int ntasks_per_socket =
        nthreads_per_socket_for_allreduce[nthreads_per_socket];
    size_t i_per_chunk = (weight_size + nsockets * CACHE_LINE_LEN - 1) /
        nsockets / CACHE_LINE_LEN * CACHE_LINE_LEN;
    size_t i_per_task =
        (i_per_chunk + ntasks_per_socket * CACHE_LINE_LEN - 1) /
        ntasks_per_socket / CACHE_LINE_LEN * CACHE_LINE_LEN;

    size_t socket_begin = sid * weight_size;
    size_t next_socket_begin = next_sid_ * weight_size;

    if (nsockets == 1) {
      int chunk_to_push = 0;
      size_t chunk_begin = std::min(chunk_to_push * i_per_chunk, weight_size);
      size_t chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

      size_t task_begin =
          std::min(chunk_begin + task_id_ * i_per_task, chunk_end);
      size_t task_end = std::min(task_begin + i_per_task, chunk_end);

      size_t src_begin = socket_begin + task_begin;

      // add reduced wgt grad to wgt
      for (size_t i = 0; i < task_end - task_begin; i += CACHE_LINE_LEN) {
        __m512 temp_v = _mm512_fmadd_ps(
            _mm512_set1_ps(-alpha_),
            _mm512_load_ps(weight_grad_->rawData() + src_begin + i),
            _mm512_load_ps(weight_->rawData() + src_begin + i));
        _mm512_store_ps(weight_->rawData() + src_begin + i, temp_v);
      }
    }

    int chunk_to_push = (idx_in_ring_ + 1 - step_ + nsockets) % nsockets;
    size_t chunk_begin = std::min(chunk_to_push * i_per_chunk, weight_size);
    size_t chunk_end = std::min(chunk_begin + i_per_chunk, weight_size);

    size_t task_begin =
        std::min(chunk_begin + task_id_ * i_per_task, chunk_end);
    size_t task_end = std::min(task_begin + i_per_task, chunk_end);

    size_t src_begin = socket_begin + task_begin;
    size_t dst_begin = next_socket_begin + task_begin;

    // add reduced wgt grad to wgt
    if (0 == step_) {
      for (size_t i = 0; i < task_end - task_begin; i += CACHE_LINE_LEN) {
        __m512 temp_v = _mm512_add_ps(
            _mm512_load_ps(weight_grad_->rawData() + src_begin + i),
            _mm512_load_ps(
                weight_grad_push_buf_->rawData() + src_begin + i));

        temp_v = _mm512_fmadd_ps(
            _mm512_set1_ps(-alpha_),
            temp_v,
            _mm512_load_ps(weight_->rawData() + src_begin + i));
        _mm512_store_ps(weight_->rawData() + src_begin + i, temp_v);
        _mm512_stream_ps(weight_->rawData() + dst_begin + i, temp_v);
      }
    } else {
      for (size_t i = 0; i < task_end - task_begin; i += CACHE_LINE_LEN) {
        _mm512_stream_ps(
            weight_->rawData() + dst_begin + i,
            _mm512_load_ps(weight_->rawData() + src_begin + i));
      }
    }

    // Make sure non-temporal stores are fully visible to other threads
    _mm_sfence();

    if (iteration_ >= NWARMUP) {
      double dt = dsecnd() - t_allgather_begin;
      double bytes = (nsockets - 1) * nrows * ncols * sizeof(float);
      int tid = numa_node_id_ * nthreads_per_socket +
          tbb::task_arena::current_thread_index();
      sum_times[tid][layer_id_][WGT_UPDATE_ALLGATHER] += dt;
      if (task_id_ == 0 && sid == 0 && step_ == 0) {
        sum_flops[layer_id_][WGT_UPDATE_ALLGATHER] += bytes;
      }
    }

#ifdef COLLECT_TRACE
    time_stamp = chrono::time_point_cast<chrono::microseconds>(
                          chrono::steady_clock::now())
                          .time_since_epoch()
                          .count();
    if (iteration_ == NWARMUP - 1) {
      traces[tid].emplace_back(Trace{time_stamp,
                                     numa_node_id_,
                                     tid - numa_node_id_ * nthreads_per_socket,
                                     string("AllGather") + "_" +
                                         to_string(nlayers - 1 - layer_id_) +
                                         "_" + to_string(task_id_),
                                     false /* end */});
    }
#endif

    ++iteration_;
  }

  void operator()(flow::continue_msg) {
    UpdateWeightAllGather::operator()();
  }

 private:
  Matrix<float, PAD>* weight_;
  Matrix<float, PAD>* weight_grad_;
  Matrix<float, PAD>* weight_grad_push_buf_;
  double alpha_;
  int layer_id_, step_, numa_node_id_;
  int task_id_; // task id within socket
  int idx_in_ring_; // my id in ring
  int next_sid_; // next socket in the ring to send data
  mutable int iteration_{0};
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

typedef flow::continue_node<flow::continue_msg> cn_type;
#ifdef USE_LIGHTWEIGHT_IN_ALL_REDUCE
typedef flow::continue_node<flow::continue_msg/*, flow::lightweight*/>
    cn_all_reduce_type;
#else
typedef flow::continue_node<flow::continue_msg> cn_all_reduce_type;
#endif

/**
 * @params prev_nodes reduce scatter step 0 will have dependencies from these
 *                    nodes
 */
void append_all_reduce_flow_graph(
    Matrix<float, PAD>* weight,
    Matrix<float, PAD>* weight_grad,
    Matrix<float, PAD>* weight_grad_push_buf,
    double alpha,
    int l, // layer
    const unique_ptr<cn_type>* prev_nodes,
    cn_type* exit,
    vector<flow::graph>& dags,
    vector<unique_ptr<flow::graph_node>>& cross_graph_edges,
    vector<unique_ptr<cn_type>>& all_reduce_first_flow_nodes,
    vector<unique_ptr<cn_all_reduce_type>>& all_reduce_flow_nodes) {
  int ntasks_per_socket =
      nthreads_per_socket_for_allreduce[nthreads_per_socket];

  // numa reduce scatter
  for (int step = 0; step < nsockets - 1; ++step) {
    for (int sid = 0; sid < nsockets; ++sid) {
      for (int task = 0; task < ntasks_per_socket; ++task) {
        int idx_in_ring, prev_sid, next_sid;
        get_my_ring_info(sid, task, &idx_in_ring, &prev_sid, &next_sid);

        if (step == 0) {
          all_reduce_first_flow_nodes.emplace_back(new cn_type(
              dags[sid],
              UpdateWeightReduceScatter(
                  weight_grad,
                  weight_grad_push_buf,
                  l,
                  step,
                  sid,
                  task,
                  idx_in_ring,
                  next_sid)));
          // Depedency from previous nodes
          make_edge(*prev_nodes[sid], *all_reduce_first_flow_nodes.back());
        } else {
          all_reduce_flow_nodes.emplace_back(new cn_all_reduce_type(
              dags[sid],
              UpdateWeightReduceScatter(
                  weight_grad,
                  weight_grad_push_buf,
                  l,
                  step,
                  sid,
                  task,
                  idx_in_ring,
                  next_sid)));

          if (step == 1) {
            // Depedency from previous step reduce scatter
            make_edge(
                *all_reduce_first_flow_nodes
                    [(l * nsockets + sid) * ntasks_per_socket + task],
                *all_reduce_flow_nodes.back());
            // Inter-socket dependency from previous step reduce scatter
            cross_graph_edges.push_back(make_crossgraph_edge(
                *all_reduce_first_flow_nodes
                    [(l * nsockets + prev_sid) * ntasks_per_socket + task],
                *all_reduce_flow_nodes.back(),
                dags[sid]));
          } else {
            // Depedency from previous step reduce scatter
            // all_reduce_flow_nodes conceptually has 4 dimensions:
            // [nlayers] x [2 * nsteps - 1] x [nsockets] x
            // [ntasks_per_socket]
            make_edge(
                *all_reduce_flow_nodes
                    [((l * (2 * (nsockets - 1) - 1) + step - 2) * nsockets +
                      sid) *
                         ntasks_per_socket +
                     task],
                *all_reduce_flow_nodes.back());
            // Inter-socket dependency from previous step reduce scatter
            cross_graph_edges.push_back(make_crossgraph_edge(
                *all_reduce_flow_nodes
                    [((l * (2 * (nsockets - 1) - 1) + step - 2) * nsockets +
                      prev_sid) *
                         ntasks_per_socket +
                     task],
                *all_reduce_flow_nodes.back(),
                dags[sid]));
          }
        }
      } // for each task
    } // for each socket
  } // for each reduce scatter step

  // numa allgather
  for (int step = 0; step < nsockets - 1; ++step) {
    for (int sid = 0; sid < nsockets; ++sid) {
      for (int task = 0; task < ntasks_per_socket; ++task) {
        int idx_in_ring, prev_sid, next_sid;
        get_my_ring_info(sid, task, &idx_in_ring, &prev_sid, &next_sid);

        all_reduce_flow_nodes.emplace_back(new cn_all_reduce_type(
            dags[sid],
            UpdateWeightAllGather(
                weight,
                weight_grad,
                weight_grad_push_buf,
                alpha,
                l,
                step,
                sid,
                task,
                idx_in_ring,
                next_sid)));
        if (nsockets == 2) {
          // Dependency from previous step
          make_edge(
              *all_reduce_first_flow_nodes
                  [(l * nsockets + sid) * ntasks_per_socket + task],
              *all_reduce_flow_nodes.back());
        } else {
          // Dependency from previous step
          make_edge(
              *all_reduce_flow_nodes
                  [((l * (2 * (nsockets - 1) - 1) + nsockets - 2 + step - 1) *
                        nsockets +
                    sid) *
                       ntasks_per_socket +
                   task],
              *all_reduce_flow_nodes.back());
          // Inter-socket dependency from previous step all gather
          cross_graph_edges.push_back(make_crossgraph_edge(
              *all_reduce_flow_nodes
                  [((l * (2 * (nsockets - 1) - 1) + nsockets - 2 + step - 1) *
                        nsockets +
                    prev_sid) *
                       ntasks_per_socket +
                   task],
              *all_reduce_flow_nodes.back(),
              dags[sid]));
        }
        if (step == nsockets - 2) {
          // Dependency to dag exit
          if (sid == 0) {
            make_edge(*all_reduce_flow_nodes.back(), *exit);
          } else {
            cross_graph_edges.push_back(make_crossgraph_edge(
                *all_reduce_flow_nodes.back(), *exit, dags[0]));
          }
        }
      } // for each task
    } // for each socket
  } // for each all gather step
}

void check_all_reduce_correctness(
    Matrix<float, PAD>* weight,
    Matrix<float, PAD>* weight_grad,
    Matrix<float, PAD>* weight_grad_push_buf) {
  int nrows = weight->nrows() / nsockets;
  int ncols = weight->ncols();

// #pragma omp parallel for collapse(2)
  for (int sid = 0; sid < nsockets; ++sid) {
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        (*weight_grad)(sid * nrows + i, j) = sid + i + j;
        (*weight)(sid * nrows + i, j) = i * j;
      }
    }
  }

  vector<flow::graph> dags(nsockets);
  for (int sid = 0; sid < nsockets; ++sid) {
    tbb_arena[sid]->execute([&dags, sid] { dags[sid].reset(); });
  }

  cn_type dag_root(
      dags[0], [&dags](flow::continue_msg) { dags[0].reserve_wait(); });
  cn_type dag_exit(
      dags[0], [&dags](flow::continue_msg) { dags[0].release_wait(); });

  vector<unique_ptr<flow::graph_node>> cross_graph_edges;
  vector<unique_ptr<cn_all_reduce_type>> all_reduce_flow_nodes;
  vector<unique_ptr<cn_type>> all_reduce_first_flow_nodes;
  vector<unique_ptr<cn_type>> prev_nodes;

  for (int sid = 0; sid < nsockets; ++sid) {
    prev_nodes.emplace_back(new cn_type(dags[sid], [](flow::continue_msg) {}));
    if (sid == 0) {
      make_edge(dag_root, *prev_nodes.back());
    } else {
      cross_graph_edges.push_back(
          make_crossgraph_edge(dag_root, *prev_nodes.back(), dags[sid]));
    }
  }

  append_all_reduce_flow_graph(
      weight,
      weight_grad,
      weight_grad_push_buf,
      1,
      0,
      &prev_nodes[0],
      &dag_exit,
      dags,
      cross_graph_edges,
      all_reduce_first_flow_nodes,
      all_reduce_flow_nodes);

  tbb_arena[0]->execute(
      [&dag_root] { dag_root.try_put(flow::continue_msg()); });
  tbb_arena[0]->execute([&dags] { dags[0].wait_for_all(); });

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

class FullyConnectedForward {
 public:
  FullyConnectedForward(
      Matrix<float, PAD>* input,
      Matrix<float, PAD>* weight,
      Matrix<float, PAD>* output,
      int layer_id,
      int numa_node_id)
      : input_(input),
        weight_(weight),
        output_(output),
        layer_id_(layer_id),
        numa_node_id_(numa_node_id) {
    iteration_ = 0;
  }

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
#ifdef SECOND_NUMA_BIND
          auto bm = numa_allocate_nodemask();
          numa_bitmask_clearall(bm);
          numa_bitmask_setbit(bm, numa_node_id_);
          numa_bind(bm);
          numa_bitmask_free(bm);
#endif

          int tid = numa_node_id_ * nthreads_per_socket +
              tbb::task_arena::current_thread_index();

#ifdef COLLECT_TRACE
          long time_stamp = chrono::time_point_cast<chrono::microseconds>(
                                chrono::steady_clock::now())
                                .time_since_epoch()
                                .count();
          if (iteration_ == NWARMUP - 1) {
            traces[tid].emplace_back(Trace{
                time_stamp,
                numa_node_id_,
                tid - numa_node_id_ * nthreads_per_socket,
                string("FC") + "_" + to_string(layer_id_) + "_" +
                    to_string(task_id),
                true /* begin */});
          }
#endif

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

#ifdef COLLECT_TRACE
          time_stamp = chrono::time_point_cast<chrono::microseconds>(
                           chrono::steady_clock::now())
                           .time_since_epoch()
                           .count();
          if (iteration_ == NWARMUP - 1) {
            traces[tid].emplace_back(Trace{
                time_stamp,
                numa_node_id_,
                tid - numa_node_id_ * nthreads_per_socket,
                string("FC") + "_" + to_string(layer_id_) + "_" +
                    to_string(task_id),
                false /* end */});
          }
#endif
        },
        tbb::simple_partitioner());

    ++iteration_;
  }

  void operator()(flow::continue_msg) {
    FullyConnectedForward::operator()();
  }

 private:
  Matrix<float, PAD>*input_, *weight_, *output_;
  int layer_id_, numa_node_id_;
  mutable int iteration_;
}; // FullyConnectedForward

class FullyConnectedBackward {
 public:
  FullyConnectedBackward(
      Matrix<float, PAD>* input,
      Matrix<float, PAD>* output_grad,
      Matrix<float, PAD>* weight,
      Matrix<float, PAD>* input_grad,
      Matrix<float, PAD>* weight_grad,
      int layer_id,
      int numa_node_id)
      : input_(input),
        output_grad_(output_grad),
        weight_(weight),
        input_grad_(input_grad),
        weight_grad_(weight_grad),
        layer_id_(layer_id),
        numa_node_id_(numa_node_id) {
    iteration_ = 0;
  }

  void operator()() const {
    // weight gradient computation
    double t0 = dsecnd();

    int m = output_grad_->ncols(), n = input_->ncols(), k = input_->nrows();
#ifdef NO_ALL_REDUCE
    int nthreads_per_socket_for_gemm = nthreads_per_socket;
#else
    int nthreads_per_socket_for_gemm = nthreads_per_socket -
        nthreads_per_socket_for_allreduce[nthreads_per_socket];
#endif

    tbb::parallel_for(
        0,
        nthreads_per_socket_for_gemm,
        [&](size_t task_id) {
#ifdef SECOND_NUMA_BIND
          auto bm = numa_allocate_nodemask();
          numa_bitmask_clearall(bm);
          numa_bitmask_setbit(bm, numa_node_id_);
          numa_bind(bm);
          numa_bitmask_free(bm);
#endif

          int tid = numa_node_id_ * nthreads_per_socket +
              tbb::task_arena::current_thread_index();

#ifdef COLLECT_TRACE
          long time_stamp = chrono::time_point_cast<chrono::microseconds>(
                                chrono::steady_clock::now())
                                .time_since_epoch()
                                .count();
          if (iteration_ == NWARMUP - 1) {
            traces[tid].emplace_back(Trace{
                time_stamp,
                numa_node_id_,
                tid - numa_node_id_ * nthreads_per_socket,
                string("FCWgtGradient") + "_" + to_string(layer_id_) + "_" +
                    to_string(task_id),
                true /* begin */});
          }
#endif

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
              nthreads_per_socket_for_gemm,
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

#ifdef COLLECT_TRACE
          time_stamp = chrono::time_point_cast<chrono::microseconds>(
                                chrono::steady_clock::now())
                                .time_since_epoch()
                                .count();
          if (iteration_ == NWARMUP - 1) {
            traces[tid].emplace_back(Trace{
                time_stamp,
                numa_node_id_,
                tid - numa_node_id_ * nthreads_per_socket,
                string("FCWgtGradient") + "_" + to_string(layer_id_) + "_" +
                    to_string(task_id),
                false /* end */});
          }
#endif
        },
        tbb::simple_partitioner());

    // backward update
    t0 = dsecnd();

    m = input_->nrows(), n = input_->ncols(), k = output_grad_->ncols();
    tbb::parallel_for(
        0,
        nthreads_per_socket_for_gemm,
        [&](size_t task_id) {
          int tid = numa_node_id_ * nthreads_per_socket +
              tbb::task_arena::current_thread_index();

#ifdef COLLECT_TRACE
          long time_stamp = chrono::time_point_cast<chrono::microseconds>(
                                chrono::steady_clock::now())
                                .time_since_epoch()
                                .count();
          if (iteration_ == NWARMUP - 1) {
            traces[tid].emplace_back(Trace{
                time_stamp,
                numa_node_id_,
                tid - numa_node_id_ * nthreads_per_socket,
                string("FCBwdGradient") + "_" + to_string(layer_id_) + "_" +
                    to_string(task_id),
                true /* begin */});
          }
#endif

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
              nthreads_per_socket_for_gemm,
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

#ifdef COLLECT_TRACE
          time_stamp = chrono::time_point_cast<chrono::microseconds>(
                                chrono::steady_clock::now())
                                .time_since_epoch()
                                .count();
          if (iteration_ == NWARMUP - 1) {
            traces[tid].emplace_back(Trace{
                time_stamp,
                numa_node_id_,
                tid - numa_node_id_ * nthreads_per_socket,
                string("FCBwdGradient") + "_" + to_string(layer_id_) + "_" +
                    to_string(task_id),
                false /* end */});
          }
#endif
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
  int layer_id_, numa_node_id_;
  mutable int iteration_;
}; // FullyConnectedBackward

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "%s <nsockets> <nthreads_per_socket>\n", argv[0]);
    exit(1);
  }

  nsockets = atoi(argv[1]);
  nthreads_per_socket = atoi(argv[2]);
  nthreads = nsockets * nthreads_per_socket;
  omp_set_num_threads(1);
  mkl_set_num_threads(1);
#ifdef COLLECT_TRACE
  traces.resize(nthreads);
#endif

  tbb::task_scheduler_init scheduler_init(nthreads);
  for (int s = 0; s < nsockets; ++s) {
    tbb_arena.emplace_back(new tbb::task_arena(nthreads_per_socket, s == 0));
    tbb_observers.emplace_back(new pinning_observer(*tbb_arena[s], s));
  }

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
#ifndef NO_ALL_REDUCE
  check_all_reduce_correctness(
      weights[0].get(),
      weight_grads[0].get(),
      weight_grad_push_bufs[0].get());
#endif

  /////////////////////////////////////////////////////////////////////////////
  // initialize values (only done by the master thread to be deterministic)
  init_matrices();

  /////////////////////////////////////////////////////////////////////////////
  // Main computation

  vector<flow::graph> dags(nsockets);
  for (int sid = 0; sid < nsockets; ++sid) {
    tbb_arena[sid]->execute([&dags, sid] { dags[sid].reset(); });
  }

  cn_type dag_root(
      dags[0], [&dags](flow::continue_msg) { dags[0].reserve_wait(); });
  cn_type dag_exit(
      dags[0], [&dags](flow::continue_msg) { dags[0].release_wait(); });

  vector<unique_ptr<cn_type>> tbb_flow_nodes;
  vector<unique_ptr<cn_all_reduce_type>> tbb_all_reduce_flow_nodes;
  vector<unique_ptr<cn_type>> tbb_all_reduce_first_flow_nodes;
  vector<unique_ptr<flow::graph_node>> cross_graph_edges;

  // forward
  for (int l = 0; l < nlayers; ++l) {
    for (int numa_node_id = 0; numa_node_id < nsockets; ++numa_node_id) {
      tbb_flow_nodes.emplace_back(new cn_type(
          dags[numa_node_id],
          FullyConnectedForward(
              activations[l].get(),
              weights[l].get(),
              activations[l + 1].get(),
              l,
              numa_node_id)));
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
      tbb_flow_nodes.emplace_back(new cn_type(
          dags[numa_node_id],
          FullyConnectedBackward(
              activations[l].get(), // input
              activations[l + 1].get(), // output_grad
              weights[l].get(),
              activations[l == 0 ? nlayers + 1 : l].get(), // input_grad
              weight_grads[l].get(),
              l,
              numa_node_id)));

      make_edge(
          *tbb_flow_nodes
              [(nlayers + (nlayers - 1 - l) - 1) * nsockets + numa_node_id],
          *tbb_flow_nodes.back());

#ifdef NO_ALL_REDUCE
      if (l == 0) {
        if (numa_node_id == 0) {
          make_edge(*tbb_flow_nodes.back(), dag_exit);
        } else {
          cross_graph_edges.push_back(
              make_crossgraph_edge(*tbb_flow_nodes.back(), dag_exit, dags[0]));
        }
      }
#endif
    } // for each socket

#ifndef NO_ALL_REDUCE
    // append flow graph for all reduce
    append_all_reduce_flow_graph(
        weights[l].get(),
        weight_grads[l].get(),
        weight_grad_push_bufs[l].get(),
        1e-10,
        nlayers - 1 - l,
        &tbb_flow_nodes[(nlayers + (nlayers - 1 - l)) * nsockets],
        &dag_exit,
        dags,
        cross_graph_edges,
        tbb_all_reduce_first_flow_nodes,
        tbb_all_reduce_flow_nodes);
#endif
  } // for each layer

  for (int it = 0; it < NWARMUP + NITER; ++it) {
    if (it == NWARMUP) {
      wall_clock_time = dsecnd();
    }
    tbb_arena[0]->execute(
        [&dag_root] { dag_root.try_put(flow::continue_msg()); });
    tbb_arena[0]->execute([&dags] { dags[0].wait_for_all(); });
  } // for each iteration
  wall_clock_time = dsecnd() - wall_clock_time;

  /////////////////////////////////////////////////////////////////////////////
  // report timing
  report_timing();

  /////////////////////////////////////////////////////////////////////////////
  // print check sum for correctness check
  print_checksum();

#ifdef COLLECT_TRACE
  // dump trace
  std::ofstream ofst("trace.log");
  ofst << "[\n";
  for (int tid = 0; tid < nthreads; ++tid) {
    for (size_t i = 0; i < traces[tid].size(); ++i) {
      traces[tid][i].serialize(ofst);
      if (tid != nthreads - 1 || i != traces[tid].size() - 1) {
        ofst << ",\n";
      }
    }
  }
  ofst << "\n]";
  ofst.close();
#endif

  return 0;
}
