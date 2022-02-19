#include "cpu_executor.h"
#include <omp.h>

namespace CpuImpl {

CpuExecutor::CpuExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): Executor(deviceStateVec, numQubits, schedule) {}
void CpuExecutor::transpose(std::vector<hptt::Transpose<cpx>> plans) { UNIMPLEMENTED(); }
void CpuExecutor::all2all(int commSize, std::vector<int> comm) { UNIMPLEMENTED(); }

#define FOLLOW_NEXT(TYPE) \
case GateType::TYPE: // no break

void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    #pragma omp parallel
    for (auto& gate: gates) {
        switch (gate.type) {
            FOLLOW_NEXT(CNOT)
            FOLLOW_NEXT(CY)
            FOLLOW_NEXT(CZ)
            FOLLOW_NEXT(CRX)
            FOLLOW_NEXT(CRY)
            FOLLOW_NEXT(CU1)
            FOLLOW_NEXT(CRZ)
            case GateType::CU: {
                int c = state.pos[gate.controlQubit];
                int t = state.pos[gate.targetQubit];
                idx_t low_bit = std::min(c, t);
                idx_t high_bit = std::max(c, t);
                idx_t mask_inner = (1 << low_bit) - 1;
                idx_t mask_middle = (1 << (high_bit - 1)) - 1 - mask_inner;
                idx_t mask_outer = (1 << (numLocalQubits - 2)) - 1 - mask_inner - mask_middle;
                #pragma omp for
                for (int i = 0; i < (1 << (numLocalQubits - 2)); i++) {
                    idx_t lo = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                    lo |= idx_t(1) << c;
                    idx_t hi = lo | (idx_t(1) << t);
                    cpx lo_val = deviceStateVec[0][lo];
                    cpx hi_val = deviceStateVec[0][hi];
                    deviceStateVec[0][lo] = lo_val * gate.mat[0][0] + hi_val * gate.mat[0][1];
                    deviceStateVec[0][hi] = lo_val * gate.mat[1][0] + hi_val * gate.mat[1][1];
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
                int t = state.pos[gate.targetQubit];
                idx_t mask_inner = (1 << t) - 1;
                idx_t mask_outer = (1 << (numLocalQubits - 1)) - 1 - mask_inner;
                #pragma omp for
                for (int i = 0; i < (1 << (numLocalQubits - 1)); i++) {
                    idx_t lo = (i & mask_inner) + ((i & mask_outer) << 1);
                    idx_t hi = lo | (idx_t(1) << t);
                    cpx lo_val = deviceStateVec[0][lo];
                    cpx hi_val = deviceStateVec[0][hi];
                    deviceStateVec[0][lo] = lo_val * gate.mat[0][0] + hi_val * gate.mat[0][1];
                    deviceStateVec[0][hi] = lo_val * gate.mat[1][0] + hi_val * gate.mat[1][1];
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
void CpuExecutor::eventBarrierAll() { UNIMPLEMENTED(); }
}