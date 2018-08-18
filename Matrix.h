#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <iomanip>
#include <iostream>
//#define USE_HUGE_PAGE
#ifdef USE_HUGE_PAGE
#include <sys/mman.h>
#endif

#include <mkl.h>

#include "Rand.h"

/// Matrix class. Stored in row-major format with
/// leading dimension equal to number of culumns
template <typename T, int PAD = 0> class Matrix {
public:
  ~Matrix() {
  }

  /// nrows x ncols matrix (uninitialized)
  Matrix(const int nrows, const int ncols, const std::string name = "")
      : nrows_(nrows), ncols_(ncols), name_(name) {
    assert(nrows_ > 0);
    assert(ncols_ > 0);
    posix_memalign(
      (void **)&data_, 4096, sizeof(T) * (nrows_ + PAD) * (ncols_ + PAD));
    assert(data_);
  }

  /// nrows x ncols matrix initialized with pre-allocated array
  Matrix(const int nrows, const int ncols, T *ptr, const std::string name = "")
      : nrows_(nrows), ncols_(ncols), name_(name) {
    assert(nrows_ > 0);
    assert(ncols_ > 0);
    data_ = ptr;
  }

  /// matrix with the given shape with uniform random data if high > low
  explicit Matrix(const std::vector<int> shape, const T low = T(0),
                  const T high = T(0), bool duplicates = true,
                  const std::string name = "")
      : nrows_(shape[0]), ncols_(shape[1]), name_(name) {
    assert(shape.size() == 2);
    assert(nrows_ > 0);
    assert(ncols_ > 0);
#ifdef USE_HUGE_PAGE
    if (size() * sizeof(T) >= 2*1024*1024) {
      //LOG(INFO) << nrows_ << " " << ncols_ << " " << size() * sizeof(T);
      data_ = (T *)mmap64(
        NULL, size() * sizeof(T), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);
      if (data_ == MAP_FAILED) {
        perror("mmap");
        assert(errno != EACCES);
        assert(errno != EAGAIN);
        assert(errno != EBADF);
        assert(errno != EINVAL);
        assert(errno != ENFILE);
        assert(errno != ENODEV);
        assert(errno != ENOMEM);
        assert(0);
      }
      madvise(data_, size() * sizeof(T), MADV_HUGEPAGE);
    }
    else
#endif
    {
      posix_memalign(
        (void **)&data_, 4096, sizeof(T) * (nrows_ + PAD) * (ncols_ + PAD));
      assert(data_);
    }
    for (int i = 0; i < nrows_; ++i) {
      for (int j = 0; j < ncols_; ++j) {
        *rawData(i, j) = low;
      }
    }
    // init randomly by sampling from the range [low, high]
    if (low < high) {
      randFill(low, high, duplicates);
    }
  }

  /// reset to new set of values
  void newInput(const T low = T(0), const T high = T(0),
                bool duplicates = true) {
    if (low < high) {
      randFill(low, high, duplicates);
    } else {
      std::fill(data_, data_ + ((size_t)nrows_ + PAD) * (ncols_ * PAD), 0);
    }
  }

  T &operator()(int row, int col) {
    assert(row >= 0 && row < nrows_);
    assert(col >= 0 && col < ncols_);
    return data_[(size_t)row * (ncols_ + PAD) + col];
  }

  const T &operator()(int row, int col) const {
    assert(row >= 0 && row < nrows_);
    assert(col >= 0 && col < ncols_);
    return data_[(size_t)row * (ncols_ + PAD) + col];
  }

  const T *rawData() const { return data_; }

  T *rawData() { return data_; }

  const T *rawData(int row, int col) const {
    return data_ + (size_t)row * (ncols_ + PAD) + col;
  }

  T *rawData(int row, int col) {
    return data_ + (size_t)row * (ncols_ + PAD) + col;
  }

  const size_t size() const { return (size_t)nrows_ * ncols_; }
  void realloc(int nrows, int ncols) {
    nrows_ = nrows;
    ncols_ = ncols;
    free(data_);
    posix_memalign(
      (void **)&data_, 4096, sizeof(T) * (nrows_ + PAD) * (ncols_ + PAD));
    assert(data_);
  }
  void realloc(const std::vector<int>& shape) {
    realloc(shape[0], shape[1]);
  }
  void assign(aligned_vector<T>& other, int nrows, int ncols) {
    nrows_ = nrows;
    ncols_ = ncols;
    assert(nrows * ncols == other.size());
    data_ = other.data();
  }

  const std::vector<int> shape() const { return {nrows_, ncols_}; }

  const std::string name() const { return name_; }

  const int nrows() const { return nrows_; }

  const int ncols() const { return ncols_; }

  int ld() const { return ncols_ + PAD; }

  bool operator==(const Matrix &other) const {
    bool eq = true;
    double rtol = 1e-6;
    for (auto i = 0; i < size(); i++) {
      if (double(std::abs(rawData()[i] - other.rawData()[i])) >
          rtol * double(std::abs(other.rawData()[i]))) {
        std::cout << rawData()[i] << " " << other.rawData()[i] << "\n";
        eq = false;
        break;
      }
    }
    return eq;
  }

  void randFill(T low, T high, bool duplicates = true) {
    for (int i = 0; i < nrows_; ++i) {
      ::randFill(rawData(i, 0), (size_t)ncols_, low, high, duplicates);
    }
  }

  T *data_;
protected:
  int nrows_;
  int ncols_;
  std::string name_;
};

#define Const(x) (x)
#define From(x) (x)
#define To(x) (x)
inline const std::vector<int> shape(int nrows, int ncols) {
  return {nrows, ncols};
}

template <typename T> T sum(Matrix<T> &m) {
  T sum = T(0);
  for (auto i = 0; i < m.size(); i++) {
    sum += m.rawData()[i];
  }
  return sum;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &m) {
  std::cout << "Matrix " << m.name() << ": " << m.nrows() << " x " << m.ncols()
            << "\n";
  for (auto i = 0; i < m.nrows(); i++) {
    for (auto j = 0; j < m.ncols(); j++) {
      os << std::setprecision(9) << m(i, j) << " ";
    }
    os << "\n";
  }
  return os;
}

#endif
