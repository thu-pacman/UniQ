#pragma once

#include <cstdio>
#include <memory>
#include <complex>

#if USE_MPI
#include <mpi.h>
#endif

#ifdef USE_DOUBLE
typedef double value_t;
typedef long long idx_t;
typedef std::complex<double> cpx;
#define MPI_Complex MPI_C_DOUBLE_COMPLEX
#else
typedef float value_t;
typedef long long idx_t;
typedef std::complex<float> cpx;
#define MPI_Complex MPI_C_COMPLEX
#endif

#define SERIALIZE_STEP(x) { *reinterpret_cast<decltype(x)*>(arr + cur) = x; cur += sizeof(x); }
#define DESERIALIZE_STEP(x) { x = *reinterpret_cast<const decltype(x)*>(arr + cur); cur += sizeof(x); }

#define SERIALIZE_VECTOR(x, result) { \
    auto tmp_chars = reinterpret_cast<const unsigned char*>(x.data()); \
    result.insert(result.end(), tmp_chars, tmp_chars + sizeof(decltype(x)::value_type) * x.size()); \
}

#define DESERIALIZE_VECTOR(x, size) { \
    x.resize(size); \
    auto tmp_size = sizeof(decltype(x)::value_type) * size; \
    memcpy(x.data(), arr + cur, tmp_size); \
    cur += tmp_size; \
}


#define UNREACHABLE() { \
    printf("file %s line %i: unreachable!\n", __FILE__, __LINE__); \
    fflush(stdout); \
    exit(1); \
}

#define UNIMPLEMENTED() { \
    printf("file %s line %i: unimplemented!\n", __FILE__, __LINE__); \
    fflush(stdout); \
    exit(1); \
}

const int LOCAL_QUBIT_SIZE = 10; // is hardcoded
const int BLAS_MAT_LIMIT = BLAS_MAT_LIMIT_DEFINED;
const int THREAD_DEP = THREAD_DEP_DEFINED; // 1 << THREAD_DEP threads per block
const int COALESCE_GLOBAL = COALESCE_GLOBAL_DEFINED;
const int MAX_GATE = 600;
const int MIN_MAT_SIZE = MIN_MAT_SIZE_DEFINED;

#define checkMPIErrors(stmt) {                          \
  int err = stmt;                                      \
  if(err != MPI_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s, line %i: %04d\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err); \
      exit(1); \
  }                                                 \
}

namespace MyGlobalVars {
    extern int numGPUs;
    extern int localGPUs;
    extern int bit;
    void init();
};

namespace MyMPI {
    extern int rank;
    extern int commSize;
    extern int commBit;
    void init();
};

template<typename T>
int bitCount(T x) {
    int ret = 0;
    for (T i = x; i; i -= i & (-i)) {
        ret++;
    }
    return ret;
}

value_t zero_wrapper(value_t x);

bool isUnitary(std::unique_ptr<cpx[]>& mat, int n);

bool operator < (const cpx& a, const cpx& b);

int get_bit(int n);