#include "cpu_executor.h"
#include <omp.h>
#include <cstring>

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