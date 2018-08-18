#include "Partition.h"

#include <iostream>
#include <omp.h>

int nsockets;

int get_num_threads_per_socket() {
  // std::cerr << "1" << std::endl;
  int nthreads =
    omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
  assert(nthreads % nsockets == 0);
  return nthreads / nsockets;
}

int get_socket_num() {
  // std::cerr << "2" << std::endl;
  int tid = omp_get_thread_num();
  return tid / get_num_threads_per_socket();
}

int get_thread_num_in_socket() {
  // std::cerr << "3" << std::endl;
  int tid = omp_get_thread_num();
  return tid % get_num_threads_per_socket();
}

int get_work_per_socket(int work) {
  return (work + nsockets - 1) / nsockets;
}

std::pair<int, int> get_partition(int work) {
  return get_partition(
    work, get_num_threads_per_socket(), get_thread_num_in_socket());
}

std::pair<int, int> get_partition(
    int work,
    int socket_id,
    int nthreads_per_socket,
    int tid_in_socket) {
  int work_per_socket = get_work_per_socket(work);
  int work_begin_socket = std::min(socket_id * work_per_socket, work);

  int work_per_thread =
      (work_per_socket + nthreads_per_socket - 1) / nthreads_per_socket;

  int work_begin = std::min(tid_in_socket * work_per_thread, work_per_socket);
  int work_end = std::min(work_begin + work_per_thread, work_per_socket);
  work_begin = std::min(work_begin_socket + work_begin, work);
  work_end = std::min(work_begin_socket + work_end, work);

  return {work_begin, work_end};
}

std::pair<int, int>
get_partition(int work, int nthreads_per_socket, int tid_in_socket) {
  return get_partition(
      work, get_socket_num(), nthreads_per_socket, tid_in_socket);
}

void get_intra_socket_2dpartition(
  int *m_begin, int *m_end, int *n_begin, int *n_end,
  int m, int n, double aspect_ratio,
  bool m_align) {

  int nthreads_per_socket = get_num_threads_per_socket();
  int tid_in_socket = get_thread_num_in_socket();

  get_intra_socket_2dpartition(
    m_begin, m_end, n_begin, n_end,
    m, n, aspect_ratio,
    m_align,
    nthreads_per_socket,
    tid_in_socket);
}

void get_intra_socket_2dpartition(
    int* m_begin,
    int* m_end,
    int* n_begin,
    int* n_end,
    int m,
    int n,
    double aspect_ratio,
    bool m_align,
    int nthreads_per_socket,
    int tid_in_socket) {
  // mb: number of thread blocks within a socket along m
  // nb: number of thread blocks along n
  // mb * nb = nthreads_per_socket
  // bm: number of rows assigned per thread block (bm = ceil(m/mb))
  // bn: number of cols assigned per thread block (bn = ceil(n/nb))
  // find mb and nb such that bm / bn is as close as possible to aspect_ratio

  int mb = 1;
  int nb = nthreads_per_socket / mb;

  int bm;
  if (m_align) {
    bm = ((m + 15) / 16 + mb - 1) / mb * 16;
  } else {
    bm = (m + mb - 1) / mb;
  }
  int bn = ((n + 15) / 16 + nb - 1) / nb * 16;

  double best_delta = std::abs((double)bm / bn - aspect_ratio);
  /*if (0 == tid) {
    VLOG(2) <<
      "bm " << bm << " bn " << bn <<
      " aspect_ratio " << aspect_ratio <<
      " best_delta " << best_delta;
  }*/

  for (int mb_candidate = 2; mb_candidate <= nthreads_per_socket;
       ++mb_candidate) {
    if (nthreads_per_socket % mb_candidate != 0)
      continue;
    int nb_candidate = nthreads_per_socket / mb_candidate;

    if (m_align) {
      if (aspect_ratio < 1 && bm == 16 &&
          (m + mb_candidate - 1) / mb_candidate < 16)
        continue;
      if ((m + mb_candidate - 1) / mb_candidate <= 8)
        continue;
    }
    if ((n + nb_candidate - 1) / nb_candidate <= 8)
      continue;

    int bm_candidate;
    if (m_align) {
      bm_candidate = ((m + 15) / 16 + mb_candidate - 1) / mb_candidate * 16;
    } else {
      bm_candidate = (m + mb_candidate - 1) / mb_candidate;
    }
    int bn_candidate = ((n + 15) / 16 + nb_candidate - 1) / nb_candidate * 16;
    double delta = std::abs((double)bm_candidate / bn_candidate - aspect_ratio);
    /*if (0 == tid) {
      VLOG(2) <<
        "bm " << bm_candidate << " bn " << bn_candidate <<
        " aspect_ratio " << aspect_ratio <<
        " delta " << delta;
    }*/

    if (delta < best_delta) {
      best_delta = delta;

      bm = bm_candidate;
      bn = bn_candidate;
      mb = mb_candidate;
      nb = nb_candidate;
    } else
      break;
  }

  /*if (0 == tid) {
    VLOG(2) << "mb " << mb << " nb " << nb << " bm " << bm << " bn " << bn;
  }*/

  int m_tid = tid_in_socket / nb;
  int n_tid = tid_in_socket % nb;

  *m_begin = std::min(m_tid * bm, m);
  *m_end = std::min(*m_begin + bm, m);

  *n_begin = std::min(n_tid * bn, n);
  *n_end = std::min(*n_begin + bn, n);
}

void get_2dpartition(
    int* m_begin,
    int* m_end,
    int* n_begin,
    int* n_end,
    int m,
    int n,
    double aspect_ratio,
    bool m_align) {
  int nthreads_per_socket = get_num_threads_per_socket();
  int tid_in_socket = get_thread_num_in_socket();

  get_2dpartition(
      m_begin,
      m_end,
      n_begin,
      n_end,
      m,
      n,
      aspect_ratio,
      m_align,
      nthreads_per_socket,
      tid_in_socket);
}

void get_2dpartition(
    int* m_begin,
    int* m_end,
    int* n_begin,
    int* n_end,
    int m,
    int n,
    double aspect_ratio,
    bool m_align,
    int socket_id,
    int nthreads_per_socket,
    int tid_in_socket) {
  int m_per_socket = get_work_per_socket(m);
  get_intra_socket_2dpartition(
      m_begin,
      m_end,
      n_begin,
      n_end,
      m_per_socket,
      n,
      aspect_ratio,
      m_align,
      nthreads_per_socket,
      tid_in_socket);

  int m_begin_socket = std::min(socket_id * m_per_socket, m);

  *m_begin = std::min(m_begin_socket + *m_begin, m);
  *m_end = std::min(m_begin_socket + *m_end, m);
}

void get_2dpartition(
    int* m_begin,
    int* m_end,
    int* n_begin,
    int* n_end,
    int m,
    int n,
    double aspect_ratio,
    bool m_align,
    int nthreads_per_socket,
    int tid_in_socket) {
  return get_2dpartition(
      m_begin,
      m_end,
      n_begin,
      n_end,
      m,
      n,
      aspect_ratio,
      m_align,
      get_socket_num(),
      nthreads_per_socket,
      tid_in_socket);
}
