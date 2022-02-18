#include "cpu/entry.h"
#include <cstring>

namespace CpuImpl {

void initState(std::vector<cpx*> &deviceStateVec, int numQubits) {
    size_t size = (sizeof(cpx) << numQubits) >> MyGlobalVars::bit;
    if ((MyGlobalVars::numGPUs > 1 && !INPLACE) || GPU_BACKEND == 3 || GPU_BACKEND == 4 || MODE > 0) {
        size <<= 1;
    }
#if INPLACE
    size += sizeof(cpx) * (1 << MAX_SLICE);
#endif
#if GPU_BACKEND == 2
    deviceStateVec.resize(1);
    deviceStateVec[0] = (cpx*) malloc(sizeof(cpx) << numQubits);
    memset(deviceStateVec[0], 0, sizeof(cpx) << numQubits);
#else
    deviceStateVec.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        deviceStateVec[g] = (cpx*) malloc(size);
        memset(deviceStateVec[g], 0, size);
    }
#endif
    cpx one(1.0);
    if  (!USE_MPI || MyMPI::rank == 0) {
        deviceStateVec[0][0] = cpx(1.0);
    }
}

void initHpttPlans(std::vector<hptt::Transpose<cpx>*>& transPlanPointers, const std::vector<int*>& transPermPointers, const std::vector<int>& locals, int numLocalQubits) {
    if (transPlanPointers.size() == 0) return;
    UNIMPLEMENTED();
}

void copyBackState(std::vector<cpx>& result, const std::vector<cpx*>& deviceStateVec, int numQubits) {
    result.resize(1ll << numQubits); // very slow ...
#if GPU_BACKEND == 0 || GPU_BACKEND == 2
    memcpy(result.data(), deviceStateVec[0], sizeof(cpx) << numQubits);
#else
    idx_t elements = 1ll << (numQubits - MyGlobalVars::bit);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        memcpy(result.data() + elements * g, deviceStateVec[g], sizeof(cpx) << (numQubits - MyGlobalVars::bit));
    }
#endif
}

void destroyState(std::vector<cpx*>& deviceStateVec) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        free(deviceStateVec[g]);
    }
}

}