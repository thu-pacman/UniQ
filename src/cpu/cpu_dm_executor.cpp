#include "cpu/cpu_dm_executor.h"
#include <assert.h>
#include <cstring>

namespace CpuImpl {

CpuDMExecutor::CpuDMExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): DMExecutor(deviceStateVec, numQubits, schedule) {}

inline void fetch_data(value_t* local_real, value_t* local_imag, const cpx* deviceStateVec, int bias, idx_t related2) {
    int x;
    unsigned int y;
    idx_t mask = (1 << (COALESCE_GLOBAL * 2)) - 1;
    assert((related2 & mask) == mask);
    related2 -= mask;
    for (x = (1 << (LOCAL_QUBIT_SIZE * 2)) - 1 - mask, y = related2; x >= 0; x -= (1 << (COALESCE_GLOBAL * 2)), y = related2 & (y-1)) {
        #pragma ivdep
        for (int i = 0; i < (1 << (COALESCE_GLOBAL * 2)); i++) {
            local_real[x + i] = deviceStateVec[(bias | y) + i].real();
            local_imag[x + i] = deviceStateVec[(bias | y) + i].imag();
            // printf("fetch %d <- %d\n", x + i, (bias | y) + i);
        }
    }
}

inline void save_data(cpx* deviceStateVec, const value_t* local_real, const value_t* local_imag, int bias, idx_t related2) {
    int x;
    unsigned int y;
    idx_t mask = (1 << (COALESCE_GLOBAL * 2)) - 1;
    assert((related2 & mask) == mask);
    related2 -= mask;
    for (x = (1 << (LOCAL_QUBIT_SIZE * 2)) - 1 - mask, y = related2; x >= 0; x -= (1 << COALESCE_GLOBAL * 2), y = related2 & (y-1)) {
        #pragma ivdep
        for (int i = 0; i < (1 << (COALESCE_GLOBAL * 2)); i++) {
            deviceStateVec[(bias | y) + i].real(local_real[x + i]);
            deviceStateVec[(bias | y) + i].imag(local_imag[x + i]);
        }
    }
}

#define CPXL(idx) (cpx(local_real[idx], local_imag[idx]))
#define CPXS(idx, val) {local_real[idx] = val.real(); local_imag[idx] = val.imag(); }

inline void apply_gate_group(value_t* local_real, value_t* local_imag, int numGates, int blockID, KernelGate hostGates[]) {
    constexpr int local2 = LOCAL_QUBIT_SIZE * 2;
    for (int i = 0; i < numGates; i++) {
        auto& gate = hostGates[i];
        int controlQubit = gate.controlQubit;
        int targetQubit = gate.targetQubit;

        if (gate.controlQubit == -1) { // single qubit gate
            // skip due to error fusion
        } else {
            if (gate.controlQubit == -3) { // two qubit gate
                controlQubit = gate.encodeQubit;
                controlQubit *= 2;
                targetQubit *= 2;
                int m = 1 << (local2 - 2);
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                for (int j = 0; j < m; j++) {
                    int s00 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s00 = ((s00 >> largeQubit) << (largeQubit + 1)) | (s00 & maskLarge);
                    int s01 = s00 | (1 << controlQubit);
                    int s10 = s00 | (1 << targetQubit);
                    int s11 = s01 | s10;

                    cpx val00 = CPXL(s00);
                    cpx val01 = CPXL(s01);
                    cpx val10 = CPXL(s10);
                    cpx val11 = CPXL(s11);

                    cpx val00_new = val00 * cpx(gate.r00, gate.i00) + val11 * cpx(gate.r11, gate.i11);
                    cpx val01_new = val01 * cpx(gate.r01, gate.i01) + val10 * cpx(gate.r10, gate.i10);
                    cpx val10_new = val01 * cpx(gate.r10, gate.i10) + val10 * cpx(gate.r01, gate.i01);
                    cpx val11_new = val00 * cpx(gate.r11, gate.i11) + val11 * cpx(gate.r00, gate.i00);

                    CPXS(s00, val00_new)
                    CPXS(s01, val01_new)
                    CPXS(s10, val10_new)
                    CPXS(s11, val11_new)
                }
                smallQubit ++; largeQubit++;
                maskSmall = (1 << smallQubit) - 1;
                maskLarge = (1 << largeQubit) - 1;

                for (int j = 0; j < m; j++) {
                    int s00 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s00 = ((s00 >> largeQubit) << (largeQubit + 1)) | (s00 & maskLarge);
                    int s01 = s00 | (1 << (controlQubit + 1));
                    int s10 = s00 | (1 << (targetQubit + 1));
                    int s11 = s01 | s10;

                    cpx val00 = CPXL(s00);
                    cpx val01 = CPXL(s01);
                    cpx val10 = CPXL(s10);
                    cpx val11 = CPXL(s11);

                    cpx val00_new = val00 * cpx(gate.r00, -gate.i00) + val11 * cpx(gate.r11, -gate.i11);
                    cpx val01_new = val01 * cpx(gate.r01, -gate.i01) + val10 * cpx(gate.r10, -gate.i10);
                    cpx val10_new = val01 * cpx(gate.r10, -gate.i10) + val10 * cpx(gate.r01, -gate.i01);
                    cpx val11_new = val00 * cpx(gate.r11, -gate.i11) + val11 * cpx(gate.r00, -gate.i00);

                    CPXS(s00, val00_new)
                    CPXS(s01, val01_new)
                    CPXS(s10, val10_new)
                    CPXS(s11, val11_new)
                }
            } else { // controlled gate
                controlQubit *= 2;
                targetQubit *= 2;
                int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                for (int j = 0; j < m; j++) {
                    int s0 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s0 = ((s0 >> largeQubit) << (largeQubit + 1)) | (s0 & maskLarge);
                    s0 |= (1 << controlQubit);
                    int s1 = s0 | (1 << targetQubit);
                    cpx val0 = CPXL(s0);
                    cpx val1 = CPXL(s1);
                    cpx val0_new = val0 * cpx(gate.r00, gate.i00) + val1 * cpx(gate.r01, gate.i01);
                    cpx val1_new = val0 * cpx(gate.r10, gate.i10) + val1 * cpx(gate.r11, gate.i11);
                    CPXS(s0, val0_new)
                    CPXS(s1, val1_new)
                }

                smallQubit ++; largeQubit++;
                maskSmall = (1 << smallQubit) - 1;
                maskLarge = (1 << largeQubit) - 1;

                for (int j = 0; j < m; j++) {
                    int s0 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s0 = ((s0 >> largeQubit) << (largeQubit + 1)) | (s0 & maskLarge);
                    s0 |= (1 << (controlQubit + 1));
                    int s1 = s0 | (1 << (targetQubit + 1));
                    cpx val0 = CPXL(s0);
                    cpx val1 = CPXL(s1);
                    cpx val0_new = val0 * cpx(gate.r00, -gate.i00) + val1 * cpx(gate.r01, -gate.i01);
                    cpx val1_new = val0 * cpx(gate.r10, -gate.i10) + val1 * cpx(gate.r11, -gate.i11);
                    CPXS(s0, val0_new)
                    CPXS(s1, val1_new)
                }

            }
            // TODO: apply error
        }
        if (hostGates[i].err_len_target > 0) {
            int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
            int qid = hostGates[i].targetQubit * 2;
            int numErrors = hostGates[i].err_len_target;
            for (int j = 0; j < m; j++) {
                int s00 = ((j >> qid) << (qid + 2)) | (j & ((1 << qid) - 1));
                int s01 = s00 | (1 << qid);
                int s10 = s00 | (1 << (qid + 1));
                int s11 = s01 | s10;
                cpx val00 = CPXL(s00);
                cpx val01 = CPXL(s01);
                cpx val10 = CPXL(s10);
                cpx val11 = CPXL(s11);

                cpx sum00 = cpx(0.0), sum01 = cpx(0.0), sum10 = cpx(0.0), sum11=cpx(0.0);
                for (int k = 0; k < numErrors; k++) {
                    cpx (*e)[2] = hostGates[i].errs_target[k];
                    cpx w00 = e[0][0] * val00 + e[0][1] * val10;
                    cpx w01 = e[0][0] * val01 + e[0][1] * val11;
                    cpx w10 = e[1][0] * val00 + e[1][1] * val10;
                    cpx w11 = e[1][0] * val01 + e[1][1] * val11;
                    sum00 += w00 * std::conj(e[0][0]) + w01 * std::conj(e[0][1]);
                    sum01 += w00 * std::conj(e[1][0]) + w01 * std::conj(e[1][1]);
                    sum10 += w10 * std::conj(e[0][0]) + w11 * std::conj(e[0][1]);
                    sum11 += w10 * std::conj(e[1][0]) + w11 * std::conj(e[1][1]);
                }

                CPXS(s00, sum00)
                CPXS(s01, sum01)
                CPXS(s10, sum10)
                CPXS(s11, sum11)
            }
        }
      , local_real[1], local_imag[1], local_real[2], local_imag[2], local_real[3], local_imag[3]);
        if (hostGates[i].err_len_control > 0) {
            int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
            int qid = hostGates[i].controlQubit == -3? hostGates[i].encodeQubit: hostGates[i].controlQubit;
            qid *= 2;
            int numErrors = hostGates[i].err_len_control;
            for (int j = 0; j < m; j++) {
                int s00 = ((j >> qid) << (qid + 2)) | (j & ((1 << qid) - 1));
                int s01 = s00 | (1 << qid);
                int s10 = s00 | (1 << (qid + 1));
                int s11 = s01 | s10;
                cpx val00 = CPXL(s00);
                cpx val01 = CPXL(s01);
                cpx val10 = CPXL(s10);
                cpx val11 = CPXL(s11);

                cpx sum00 = cpx(0.0), sum01 = cpx(0.0), sum10 = cpx(0.0), sum11=cpx(0.0);
                for (int k = 0; k < numErrors; k++) {
                    cpx (*e)[2] = hostGates[i].errs_control[k];
                    cpx w00 = e[0][0] * val00 + e[0][1] * val10;
                    cpx w01 = e[0][0] * val01 + e[0][1] * val11;
                    cpx w10 = e[1][0] * val00 + e[1][1] * val10;
                    cpx w11 = e[1][0] * val01 + e[1][1] * val11;
                    sum00 += w00 * std::conj(e[0][0]) + w01 * std::conj(e[0][1]);
                    sum01 += w00 * std::conj(e[1][0]) + w01 * std::conj(e[1][1]);
                    sum10 += w10 * std::conj(e[0][0]) + w11 * std::conj(e[0][1]);
                    sum11 += w10 * std::conj(e[1][0]) + w11 * std::conj(e[1][1]);
                }

                CPXS(s00, sum00)
                CPXS(s01, sum01)
                CPXS(s10, sum10)
                CPXS(s11, sum11)
            }
        }
    }
}

void CpuDMExecutor::launchPerGateGroupDM(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    cpx* sv = deviceStateVec[0];
    unsigned int related2 = duplicate_bit(relatedQubits); // warning: put it in master core
    idx_t blockHot = (idx_t(1) << (numLocalQubits * 2)) - 1 - related2;
    #pragma omp parallel for
    for (int blockID = 0; blockID < (1 << ((numLocalQubits - LOCAL_QUBIT_SIZE) * 2)); blockID++) {
        value_t local_real[1 << (LOCAL_QUBIT_SIZE * 2)];
        value_t local_imag[1 << (LOCAL_QUBIT_SIZE * 2)];
        unsigned int bias = 0;
        {
            int bid = blockID;
            for (unsigned int bit = 1; bit < (1u << (numLocalQubits * 2)); bit <<= 1) {
                if (blockHot & bit) {
                    if (bid & 1)
                        bias |= bit;
                    bid >>= 1;
                }
            }
        }
        fetch_data(local_real, local_imag, sv, bias, related2);
        apply_gate_group(local_real, local_imag, gates.size(), blockID, hostGates);
        save_data(sv, local_real, local_imag, bias, related2);
    }
}

void CpuDMExecutor::dm_transpose() { UNIMPLEMENTED(); }

void CpuDMExecutor::transpose(std::vector<std::shared_ptr<hptt::Transpose<cpx>>> plans) {
    plans[0]->setInputPtr(deviceStateVec[0]);
    plans[0]->setOutputPtr(deviceBuffer[0]);
    plans[0]->execute();
}

void CpuDMExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) {
    int numLocalQubits = numQubits - MyGlobalVars::bit / 2;
    idx_t oldGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        oldGlobals |= 1ll << state.layout[i];
    idx_t newGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        newGlobals |= 1ll << newState.layout[i];
    
    idx_t localMasks[commSize];
    idx_t localMask = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        if (newState.layout[i] != state.layout[i]) {
            int x = state.layout[i];
            localMask |= 1ll << newState.pos[x];
        }
    

    idx_t sliceSize = 0;
    while (sliceSize < MAX_SLICE && !(localMask >> sliceSize & 1))
        sliceSize ++;
    sliceSize = idx_t(1) << (sliceSize * 2);

    localMask = duplicate_bit(localMask);
    for (idx_t i = commSize-1, msk = localMask; i >= 0; i--, msk = localMask & (msk - 1)) {
        localMasks[i] = msk;
    }

    cpx* tmpBuffer[MyGlobalVars::localGPUs];
    size_t tmpStart = 1ll << (numLocalQubits * 2);
    if (GPU_BACKEND == 3 || GPU_BACKEND == 4)
        tmpStart <<= 1;
    for (int i = 0; i < MyGlobalVars::localGPUs; i++)
        tmpBuffer[i] = deviceStateVec[i] + tmpStart;

    for (idx_t iter = 0; iter < (1ll << (numLocalQubits * 2)); iter += sliceSize) {
        if (iter & localMask) continue;
        for (int xr = 1; xr < commSize; xr++) {
            // copy from src to tmp_buffer
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                // the (a%commSize)-th GPU in the a/commSize comm_world (comm[a]) ->
                // the (a%commSize)^xr-th GPU in the same comm_world comm[a^xr]
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t srcBias = iter + localMasks[b % commSize];
#if USE_MPI
                int comm_a = comm[a] %  MyGlobalVars::localGPUs;
                checkMPIErrors(MPI_Sendrecv(
                    deviceStateVec[comm_a] + srcBias, sliceSize, MPI_Complex, comm[b], 0,
                    tmpBuffer[comm_a], sliceSize, MPI_Complex, comm[b], MPI_ANY_TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                ));
#else
                UNREACHABLE();
#endif
            }
            // copy from tmp_buffer to dst
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[b] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t dstBias = iter + localMasks[a % commSize];
                int comm_b = comm[b] % MyGlobalVars::localGPUs;
                memcpy(deviceStateVec[comm_b] + dstBias, tmpBuffer[comm_b], (sizeof(cpx) * sliceSize));
            }
        }
    }
}

void CpuDMExecutor::all2all(int commSize, std::vector<int> comm) {
    int numLocalQubit = numQubits - MyGlobalVars::bit / 2;
    idx_t numElements = 1ll << (numLocalQubit * 2);
    int numPart = numSlice / commSize;
    idx_t partSize = numElements / numSlice;
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
            }
        }
    }
}

void CpuDMExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) { UNIMPLEMENTED(); }
void CpuDMExecutor::launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuDMExecutor::launchBlasGroup(GateGroup& gg, int numLocalQubits) { UNIMPLEMENTED(); }
void CpuDMExecutor::launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }

void CpuDMExecutor::deviceFinalize() {}

void CpuDMExecutor::sliceBarrier(int sliceID) { UNIMPLEMENTED(); }
void CpuDMExecutor::eventBarrier() { UNIMPLEMENTED(); }
void CpuDMExecutor::eventBarrierAll() {
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

void CpuDMExecutor::allBarrier() {
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

}