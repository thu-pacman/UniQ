#include "cpu_executor.h"
#include <omp.h>
#include <cstring>
#include <assert.h>

namespace CpuImpl {

CpuExecutor::CpuExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): Executor(deviceStateVec, numQubits, schedule) {}

void CpuExecutor::transpose(std::vector<std::shared_ptr<hptt::Transpose<cpx>>> plans) {
    plans[0]->setInputPtr(deviceStateVec[0]);
    plans[0]->setOutputPtr(deviceBuffer[0]);
    plans[0]->execute();
}

void CpuExecutor::all2all(int commSize, std::vector<int> comm) {
    int numLocalQubit = numQubits - MyGlobalVars::bit;
    idx_t numElements = 1ll << numLocalQubit;
    int numPart = numSlice / commSize;
    idx_t partSize = numElements / numSlice;
    partID.resize(numSlice * MyGlobalVars::localGPUs);
    peer.resize(numSlice * MyGlobalVars::localGPUs);
    int sliceID = 0;
    for (int xr = 0; xr < commSize; xr++) {
        for (int p = 0; p < numPart; p++) {
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                int comm_a = comm[a] % MyGlobalVars::localGPUs;
                int srcPart = a % commSize * numPart + p;
                int dstPart = b % commSize * numPart + p;
#if USE_MPI
                if (a == b) {
                    memcpy(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        deviceBuffer[comm_a] + srcPart * partSize,
                        partSize * sizeof(cpx)
                    );
                } else {
                    checkMPIErrors(MPI_Sendrecv(
                        deviceBuffer[comm_a] + dstPart * partSize, partSize, MPI_Complex, comm[b], 0,
                        deviceStateVec[comm_a] + dstPart * partSize, partSize, MPI_Complex, comm[b], MPI_ANY_TAG,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE
                    ));
                }
#else
                UNIMPLEMENTED();
#endif
                partID[sliceID * MyGlobalVars::localGPUs + comm_a] = dstPart;
                peer[sliceID * MyGlobalVars::localGPUs + comm_a] = comm[b];
            }
            // events should be recorded after ncclGroupEnd
#ifdef ENABLE_OVERLAP
            
            UNIMPLEMENTED();
#endif
            sliceID++;
        }
    }
#ifndef ENABLE_OVERLAP
    this->eventBarrierAll();
#endif
}

#define FOLLOW_NEXT(TYPE) \
case GateType::TYPE: // no break

#if GPU_BACKEND==1

inline void fetch_data(cpx* local, const cpx* deviceStateVec, int bias, idx_t relatedQubits) {
    int x;
    unsigned int y;
    for (x = ((1 << LOCAL_QUBIT_SIZE) - 1), y = relatedQubits; x >= 0; x--, y = relatedQubits & (y-1)) {
        local[x] = deviceStateVec[bias | y];
    }
}

inline void save_data(cpx* deviceStateVec, const cpx* local, int bias, idx_t relatedQubits) {
    int x;
    unsigned int y;
    for (x = ((1 << LOCAL_QUBIT_SIZE) - 1), y = relatedQubits; x >= 0; x--, y = relatedQubits & (y-1)) {
        deviceStateVec[bias | y] = local[x];
    }
}

inline void apply_gate_group(cpx* local, int numGates, int blockID, KernelGate hostGates[]) {
    for (int i = 0; i < numGates; i++) {
        auto& gate = hostGates[i];
        int controlQubit = gate.controlQubit;
        int targetQubit = gate.targetQubit;
        char controlIsGlobal = gate.controlIsGlobal;
        char targetIsGlobal = gate.targetIsGlobal;
        if (!controlIsGlobal) {
            if (!targetIsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 2);
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                // TODO: switch (gate)
                for (int j = 0; j < m; j++) {
                    int lo = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    lo = ((lo >> largeQubit) << (largeQubit + 1)) | (lo & maskLarge);
                    lo |= 1 << controlQubit;
                    int hi = lo | (1 << targetQubit);
                    cpx lo_val = local[lo];
                    cpx hi_val = local[hi];
                    local[lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    local[hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
            } else {
                assert(hostGates[i].type == GateType::CZ || hostGates[i].type == GateType::CU1 || hostGates[i].type == GateType::CRZ);
                bool isHighBlock = (blockID >> targetQubit) & 1;
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                int maskControl = (1 << controlQubit) - 1;
                if (!isHighBlock){
                    if (hostGates[i].type == GateType::CRZ) {
                        for (int j = 0; j < m; j++) {
                            int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                            local[x] = local[x] * cpx(gate.r00, gate.i00);
                        }
                    }
                } else {
                    // TODO: switch (gate)
                    for (int j = 0; j < m; j++) {
                        int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                        local[x] = local[x] * cpx(gate.r11, gate.i11);
                    }
                }
            }
        } else {
            if (controlIsGlobal == 1 && !((blockID>> controlQubit) & 1)) {
                continue;
            }
            if (!targetIsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                int maskTarget = (1 << targetQubit) - 1;
                // TODO: switch (gate)
                for (int j = 0; j < m; j++) {
                    int lo = ((j >> targetQubit) << (targetQubit + 1)) | (j & maskTarget);
                    int hi = lo | (1 << targetQubit);
                    cpx lo_val = local[lo];
                    cpx hi_val = local[hi];
                    local[lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    local[hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
            } else {
                bool isHighBlock = (blockID >> targetQubit) & 1;
                // TODO: switch (gate)
                int m = 1 << LOCAL_QUBIT_SIZE;
                if (!isHighBlock){ \
                    for (int j = 0; j < m; j++) {
                        local[j] = local[j] * cpx(gate.r00, gate.i00);
                    }
                } else {
                    for (int j = 0; j < m; j++) {
                        local[j] = local[j] * cpx(gate.r11, gate.i11);
                    }
                }
            }
        }
    }
}

void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    cpx* sv = deviceStateVec[0];
    #pragma omp parallel for
    for (int blockID = 0; blockID < (1 << (numLocalQubits - LOCAL_QUBIT_SIZE)); blockID++) {
        cpx local[1 << LOCAL_QUBIT_SIZE];
        idx_t blockHot = (idx_t(1) << numLocalQubits) - 1 - relatedQubits;
        unsigned int bias = 0;
        {
            int bid = blockID;
            for (unsigned int bit = 1; bit < (1u << numLocalQubits); bit <<= 1) {
                if (blockHot & bit) {
                    if (bid & 1)
                        bias |= bit;
                    bid >>= 1;
                }
            }
        }
        fetch_data(local, sv, bias, relatedQubits);
        apply_gate_group(local, gates.size(), blockID, hostGates);
        save_data(sv, local, bias, relatedQubits);
    }
}
#elif GPU_BACKEND==2
void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    #pragma omp parallel
    for (int i = 0; i < int(gates.size()); i++) {
        auto& gate = hostGates[i];
        switch (gate.type) {
            FOLLOW_NEXT(CNOT)
            FOLLOW_NEXT(CY)
            FOLLOW_NEXT(CZ)
            FOLLOW_NEXT(CRX)
            FOLLOW_NEXT(CRY)
            FOLLOW_NEXT(CU1)
            FOLLOW_NEXT(CRZ)
            case GateType::CU: {
                int c = gate.controlQubit;
                int t = gate.targetQubit;
                idx_t low_bit = std::min(c, t);
                idx_t high_bit = std::max(c, t);
                idx_t mask_inner = (idx_t(1) << low_bit) - 1;
                idx_t mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
                idx_t mask_outer = (idx_t(1) << (numLocalQubits - 2)) - 1 - mask_inner - mask_middle;
                #pragma omp for
                for (idx_t i = 0; i < (idx_t(1) << (numLocalQubits - 2)); i++) {
                    idx_t lo = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                    lo |= idx_t(1) << c;
                    idx_t hi = lo | (idx_t(1) << t);
                    cpx lo_val = deviceStateVec[0][lo];
                    cpx hi_val = deviceStateVec[0][hi];
                    deviceStateVec[0][lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    deviceStateVec[0][hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
                break;
            }
            FOLLOW_NEXT(U1)
            FOLLOW_NEXT(U2)
            FOLLOW_NEXT(U3)
            FOLLOW_NEXT(U)
            FOLLOW_NEXT(H)
            FOLLOW_NEXT(X)
            FOLLOW_NEXT(Y)
            FOLLOW_NEXT(Z)
            FOLLOW_NEXT(S)
            FOLLOW_NEXT(SDG)
            FOLLOW_NEXT(T)
            FOLLOW_NEXT(TDG)
            FOLLOW_NEXT(RX)
            FOLLOW_NEXT(RY)
            FOLLOW_NEXT(RZ)
            FOLLOW_NEXT(ID)
            FOLLOW_NEXT(GII)
            FOLLOW_NEXT(GZZ)
            FOLLOW_NEXT(GOC)
            case GateType::GCC: {
                int t = gate.targetQubit;
                idx_t mask_inner = (idx_t(1) << t) - 1;
                idx_t mask_outer = (idx_t(1) << (numLocalQubits - 1)) - 1 - mask_inner;
                #pragma omp for
                for (idx_t i = 0; i < (idx_t(1) << (numLocalQubits - 1)); i++) {
                    idx_t lo = (i & mask_inner) + ((i & mask_outer) << 1);
                    idx_t hi = lo | (idx_t(1) << t);
                    cpx lo_val = deviceStateVec[0][lo];
                    cpx hi_val = deviceStateVec[0][hi];
                    deviceStateVec[0][lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    deviceStateVec[0][hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
                break;
            }
            default: {
                UNIMPLEMENTED();
            }
            #pragma omp barrier
        }
    }
}
#else
void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    UNIMPLEMENTED()
}
#endif

void CpuExecutor::deviceFinalize() {}

void CpuExecutor::allBarrier() {
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}


void CpuExecutor::dm_transpose()  { UNIMPLEMENTED(); }
void CpuExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) { UNIMPLEMENTED(); }
void CpuExecutor::launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::launchBlasGroup(GateGroup& gg, int numLocalQubits) { UNIMPLEMENTED(); }
void CpuExecutor::launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::sliceBarrier(int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::eventBarrier() { UNIMPLEMENTED(); }
void CpuExecutor::eventBarrierAll() { checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD)); }
}