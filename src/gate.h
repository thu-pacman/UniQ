#pragma once

#include <string>
#include <vector>
#include "utils.h"

enum class GateType {
    CNOT, CY, CZ, CRX, CRY, CU1, CRZ, CU, U1, U2, U3, U, H, X, Y, Z, S, SDG, T, TDG, RX, RY, RZ, RZZ, MCU, TOTAL, ID, GII, GZZ, GOC, GCC, DIG, MCI, V01
};

struct Error {
    Error() = default;
    Error(GateType type_, cpx mat00_, cpx mat01_, cpx mat10_, cpx mat11_):
        type(type_), mat00(mat00_), mat01(mat01_), mat10(mat10_), mat11(mat11_) {}
    GateType type;
    cpx mat00, mat01, mat10, mat11;
};

struct Gate {
    int gateID;
    GateType type;
    cpx mat[2][2];
    std::string name;
    int targetQubit;
    int controlQubit; // -1 for single bit gate， -2 for MC gates, -3 for two qubit gates
    idx_t encodeQubit; // bit map of the control qubits of MC gates, target2 for two qubit gate
    std::vector<int> controlQubits;
    std::vector<Error> controlErrors;
    std::vector<Error> targetErrors;
    Gate(): controlQubit(-1), encodeQubit(0) {};
    Gate(const Gate&) = default;
    bool isControlGate() const {
        return controlQubit >= 0;
    }
    bool isSingleGate() const {
        return controlQubit == -1;
    }
    bool isMCGate() const {
        return controlQubit == -2;
    }
    bool isTwoQubitGate() const {
        return controlQubit == -3;
    }
#if MODE == 2
    bool isDiagonal() const { return false; }
#else
    bool isDiagonal() const {
        return type == GateType::CZ || type == GateType::CU1 || type == GateType::CRZ || type == GateType::U1 || type == GateType::Z || type == GateType::S || type == GateType::SDG || type == GateType::T || type == GateType::TDG || type == GateType::RZ || type == GateType::RZZ || type == GateType::DIG;
    }
#endif
    bool hasControl(int q) const {
        if (isControlGate()) return controlQubit == q;
        if (isMCGate()) return encodeQubit >> q & 1;
        return false;
    }
    bool hasTarget(int q) const {
        if (isTwoQubitGate()) return targetQubit == q || encodeQubit == q;
        return targetQubit == q;
    }
    static Gate CNOT(int controlQubit, int targetQubit);
    static Gate CY(int controlQubit, int targetQubit);
    static Gate CZ(int controlQubit, int targetQubit);
    static Gate CRX(int controlQubit, int targetQubit, value_t angle);
    static Gate CRY(int controlQubit, int targetQubit, value_t angle);
    static Gate CU1(int controlQubit, int targetQubit, value_t lambda);
    static Gate CRZ(int controlQubit, int targetQubit, value_t angle);
    static Gate CU(int controlQubit, int targetQubit, std::vector<cpx> params);
    static Gate U1(int targetQubit, value_t lambda);
    static Gate U2(int targetQubit, value_t phi, value_t lambda);
    static Gate U3(int targetQubit, value_t theta, value_t phi, value_t lambda);
    static Gate U(int targetQubit, std::vector<cpx> params);
    static Gate H(int targetQubit);
    static Gate X(int targetQubit);
    static Gate Y(int targetQubit);
    static Gate Z(int targetQubit);
    static Gate S(int targetQubit);
    static Gate SDG(int targetQubit); 
    static Gate T(int targetQubit);
    static Gate TDG(int targetQubit);
    static Gate RX(int targetQubit, value_t angle);
    static Gate RY(int targetQubit, value_t angle);
    static Gate RZ(int targetQubit, value_t angle);
    static Gate ID(int targetQubit);
    static Gate GII(int targetQubit);
    static Gate GTT(int targetQubit);
    static Gate GZZ(int targetQubit);
    static Gate GOC(int targetQubit, value_t r, value_t i);
    static Gate GCC(int targetQubit, value_t r, value_t i);
    static Gate DIG(int targetQubit, cpx lo, cpx hi);
    static Gate V01(int targetQubit, cpx val);
    static Gate RZZ(int targetQubit1, int targetQubit2, value_t angle);
    static Gate MCU(std::vector<int> controlQubits, int targetQubit, std::vector<cpx> params);
    static Gate random(int lo, int hi);
    static Gate random(int lo, int hi, GateType type);
    static Gate control(int controlQubit, int targetQubit, GateType type);
    static GateType toU(GateType type);
    static std::string get_name(GateType ty);
    std::vector<unsigned char> serialize() const;
    static Gate deserialize(const unsigned char* arr, int& cur);
};

struct KernelGate {
    int targetQubit;
    int controlQubit;
    idx_t encodeQubit;
    GateType type;
    char targetIsGlobal;  // 0-local 1-global
    char controlIsGlobal; // 0-local 1-global 2-not control 
    value_t r00, i00, r01, i01, r10, i10, r11, i11;

#if MODE == 2
    int err_len_control, err_len_target;
    cpx errs_control[MAX_ERROR_LEN][2][2]; // channel, mat_row, mat_col, real/imag
    cpx errs_target[MAX_ERROR_LEN][2][2];
    KernelGate(
        GateType type_,
        idx_t encodeQubit_, 
        int controlQubit_, char controlIsGlobal_,
        int targetQubit_, char targetIsGlobal_,
        const cpx mat[2][2]
    ):
        targetQubit(targetQubit_), controlQubit(controlQubit_), encodeQubit(encodeQubit_),
        type(type_),
        targetIsGlobal(targetIsGlobal_), controlIsGlobal(controlIsGlobal_),
        r00(mat[0][0].real()), i00(mat[0][0].imag()), r01(mat[0][1].real()), i01(mat[0][1].imag()),
        r10(mat[1][0].real()), i10(mat[1][0].imag()), r11(mat[1][1].real()), i11(mat[1][1].imag()),
        err_len_control(0), err_len_target(0) {}

#else
    KernelGate(
        GateType type_,
        idx_t encodeQubit_, 
        int controlQubit_, char controlIsGlobal_,
        int targetQubit_, char targetIsGlobal_,
        const cpx mat[2][2]
    ):
        targetQubit(targetQubit_), controlQubit(controlQubit_), encodeQubit(encodeQubit_),
        type(type_),
        targetIsGlobal(targetIsGlobal_), controlIsGlobal(controlIsGlobal_),
        r00(mat[0][0].real()), i00(mat[0][0].imag()), r01(mat[0][1].real()), i01(mat[0][1].imag()),
        r10(mat[1][0].real()), i10(mat[1][0].imag()), r11(mat[1][1].real()), i11(mat[1][1].imag()) {}
#endif

    KernelGate() = default;

    // controlled gate
    static KernelGate controlledGate(
        GateType type,
        int controlQubit, char controlIsGlobal,
        int targetQubit, char targetIsGlobal,
        const cpx mat[2][2]
    ) {
        return KernelGate(type, 0, controlQubit, controlIsGlobal, targetQubit, targetIsGlobal, mat);
    }

    // multi-control gate
    static KernelGate mcGate(
        GateType type,
        idx_t mcQubits,
        int targetQubit, char targetIsGlobal,
        const cpx mat[2][2]
    ) {
        return KernelGate(type, mcQubits, -2, 2, targetQubit, targetIsGlobal, mat);
    }

    // two qubit gate
    static KernelGate twoQubitGate(
        GateType type,
        int targetQubit1, char target1IsGlobal,
        int targetQubit2, char target2IsGlobal,
        const cpx mat[2][2]
    ) {
        return KernelGate(type, targetQubit1, -3, target1IsGlobal, targetQubit2, target2IsGlobal, mat);
    }

    // single qubit gate
    static KernelGate singleQubitGate(
        GateType type,
        int targetQubit, char targetIsGlobal,
        const cpx mat[2][2]
    ) {
        return KernelGate(type, 0, -1, -1, targetQubit, targetIsGlobal, mat);
    }

    static KernelGate ID() {
        cpx mat[2][2] = {1, 0, 0, 1}; \
        return KernelGate::singleQubitGate(GateType::ID, 0, 0, mat);
    }
};