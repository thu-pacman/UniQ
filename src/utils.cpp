#include "utils.h"

#include <cstring>
#include "logger.h"

#ifdef USE_GPU
#include "cuda/entry.h"
#elif  USE_CPU
#include "cpu/entry.h"
#endif

namespace MyGlobalVars {
int numGPUs;
int localGPUs;
int bit;

void init() {
#ifdef USE_GPU
    CudaImpl::initCudaObjects();
#else
    localGPUs = 1;
    #if USE_MPI
        numGPUs = MyMPI::commSize * localGPUs;
    #else
        numGPUs = localGPUs;
    #endif
    Logger::add("Local GPU: %d", localGPUs);
    bit = get_bit(numGPUs);
#endif
#if USE_CPU
    CpuImpl::initCpu();
#endif
}
};

namespace MyMPI {
int rank;
int commSize;
int commBit;
void init() {
#if USE_MPI
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
#endif
}
};


value_t zero_wrapper(value_t x) {
    const value_t eps = 1e-14;
    if (x > -eps && x < eps) {
        return 0;
    } else {
        return x;
    }
}

bool isUnitary(std::unique_ptr<cpx[]>& mat, int n) {
    cpx result[n * n];
    // memset(result, 0, sizeof(result));
    for (int k = 0; k < n; k++)
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                cpx v1 = std::conj(mat[k * n + i]);
                result[i * n + j] = result[i * n + j] + v1 * mat[k * n + j];
            }
    bool wa = 0;
    value_t eps = 1e-8;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        cpx val = result[i * n + i];
        if (fabs(val.real() - 1) > eps || fabs(val.imag()) > eps) {
            wa = 1;
        }
        for (int j = 0; j < n; j++) {
            if (i == j)
                continue;
            cpx val = result[i * n + j];
            if (fabs(val.real()) > eps || fabs(val.imag()) > eps)
                wa = 1;
        }
    }
    if (wa) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("(%.2f %.2f) ", result[i * n + j].real(), result[i * n + j].imag());
            printf("\n");
        }
        exit(1);
    }
    return 1;
}

bool operator < (const cpx& a, const cpx& b) {
        return a.real() == b.real() ? a.imag() < b.imag() : a.real() < b.real();
}

int get_bit(int n) {
    int x = n;
    int bit = -1;
    while (x) {
        bit ++;
        x >>= 1;
    }
    if (n == 0 || (1 << bit) != n) {
        printf("Must be pow of two: %d\n", n);
        exit(1);
    }
    return bit;
}

idx_t to_bitmap(std::vector<int> qubits) {
    idx_t ret;
    for (auto& x: qubits) ret |= 1ll << x;
    return ret;
}
