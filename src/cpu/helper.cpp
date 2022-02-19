#include "cpu/entry.h"
#include <cstring>
#include <memory>
#include "hptt.h"

namespace MyGlobalVars {
    int n_thread;
}

namespace CpuImpl {

void initCpu() {
    #pragma omp parallel
    {
        #pragma omp master
        MyGlobalVars::n_thread = omp_get_num_threads();
    }
}

void initState(std::vector<cpx*> &deviceStateVec, int numQubits) {
    size_t size = (sizeof(cpx) << numQubits) >> MyGlobalVars::bit;
    if ((MyGlobalVars::numGPUs > 1 && !INPLACE) || GPU_BACKEND == 3 || GPU_BACKEND == 4 || MODE > 0) {
        size <<= 1;
    }
#if INPLACE
    size += sizeof(cpx) * (1 << MAX_SLICE);
#endif
    deviceStateVec.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        deviceStateVec[g] = (cpx*) malloc(size);
        memset(deviceStateVec[g], 0, size);
    }
    cpx one(1.0);
    if  (!USE_MPI || MyMPI::rank == 0) {
        deviceStateVec[0][0] = cpx(1.0);
    }
}

void initHpttPlans(std::vector<std::shared_ptr<hptt::Transpose<cpx>>*>& transPlanPointers, const std::vector<int*>& transPermPointers, const std::vector<int>& locals, int numLocalQubits) {
    if (transPlanPointers.size() == 0) return;
    int total = transPlanPointers.size();
    std::vector<int> dims(numLocalQubits, 2);
    for (int i = 0; i < total; i++) {
        *transPlanPointers[i] = hptt::create_plan(
            transPermPointers[i], numLocalQubits,
            cpx(1.0), nullptr, dims.data(), nullptr,
            cpx(0.0), nullptr, nullptr,
            hptt::ESTIMATE, MyGlobalVars::n_thread
        );
    }
}

void copyBackState(std::vector<cpx>& result, const std::vector<cpx*>& deviceStateVec, int numQubits) {
    result.resize(1ll << numQubits); // very slow ...
    idx_t elements = 1ll << (numQubits - MyGlobalVars::bit);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        memcpy(result.data() + elements * g, deviceStateVec[g], sizeof(cpx) << (numQubits - MyGlobalVars::bit));
    }
}

void destroyState(std::vector<cpx*>& deviceStateVec) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        free(deviceStateVec[g]);
    }
}

}