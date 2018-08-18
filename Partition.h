#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

extern int nsockets;

int get_num_threads_per_socket();
int get_socket_num();
int get_thread_num_in_socket();

int get_work_per_socket(int work);
std::pair<int, int> get_partition(int work);
std::pair<int, int>
get_partition(int work, int nthreads_per_socket, int tid_in_socket);
std::pair<int, int> get_partition(
    int work,
    int socket_id,
    int nthreads_per_socket,
    int tid_in_socket);

void get_intra_socket_2dpartition(
    int* m_begin,
    int* m_end,
    int* n_begin,
    int* n_end,
    int m,
    int n,
    double aspect_ratio,
    bool m_align);
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
    int tid_in_socket);

void get_2dpartition(
    int* m_begin,
    int* m_end,
    int* n_begin,
    int* n_end,
    int m,
    int n,
    double aspect_ratio,
    bool m_align);
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
    int tid_in_socket);
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
    int tid_in_socket);
