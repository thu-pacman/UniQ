#include "cpu_executor.h"

namespace CpuImpl {

CpuExecutor::CpuExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): Executor(deviceStateVec, numQubits, schedule) {}
void CpuExecutor::transpose(std::vector<hptt::Transpose<cpx>> plans) { UNIMPLEMENTED(); }
void CpuExecutor::all2all(int commSize, std::vector<int> comm) { UNIMPLEMENTED(); }

void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits) {
    printf("[warning] not apply gates\n");
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