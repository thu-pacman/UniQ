#include "gate.h"

#include <cmath>
#include <cstring>
#include <assert.h>

static int globalGateID = 0;

Gate Gate::CNOT(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CNOT;
    g.mat[0][0] = cpx(0); g.mat[0][1] = cpx(1);
    g.mat[1][0] = cpx(1); g.mat[1][1] = cpx(0);
    g.name = "CN";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CY(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CY;
    g.mat[0][0] = cpx(0); g.mat[0][1] = cpx(0, -1);
    g.mat[1][0] = cpx(0, 1); g.mat[1][1] = cpx(0);
    g.name = "CY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CZ(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CZ;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(-1);
    g.name = "CZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRX(int controlQubit, int targetQubit, value_t angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRX;
    g.mat[0][0] = cpx(cos(angle/2.0)); g.mat[0][1] = cpx(0, -sin(angle/2.0));
    g.mat[1][0] = cpx(0, -sin(angle/2.0)); g.mat[1][1] = cpx(cos(angle/2.0));
    g.name = "CRX";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRY(int controlQubit, int targetQubit, value_t angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRY;
    g.mat[0][0] = cpx(cos(angle/2.0)); g.mat[0][1] = cpx(-sin(angle/2.0));
    g.mat[1][0] = cpx(sin(angle/2.0)); g.mat[1][1] = cpx(cos(angle/2.0));
    g.name = "CRY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CU1(int controlQubit, int targetQubit, value_t lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CU1;
    g.mat[0][0] = cpx(1);
    g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0);
    g.mat[1][1] = cpx(cos(lambda), sin(lambda));
    g.name = "CU1";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRZ(int controlQubit, int targetQubit, value_t angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRZ;
    g.mat[0][0] = cpx(cos(angle/2), -sin(angle/2)); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(cos(angle/2), sin(angle/2));
    g.name = "CRZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CU(int controlQubit, int targetQubit, std::vector<cpx> params) {
    Gate g;
    g.gateID = ++globalGateID;
    g.type = GateType::CU;
    g.mat[0][0] = params[0]; g.mat[0][1] = params[1];
    g.mat[1][0] = params[2]; g.mat[1][1] = params[3];
    g.name = "CU";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::U1(int targetQubit, value_t lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U1;
    g.mat[0][0] = cpx(1);
    g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0);
    g.mat[1][1] = cpx(cos(lambda), sin(lambda));
    g.name = "U1";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U2(int targetQubit, value_t phi, value_t lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U2;
    g.mat[0][0] = cpx(1.0 / sqrt(2));
    g.mat[0][1] = cpx(-cos(lambda) / sqrt(2), -sin(lambda) / sqrt(2));
    g.mat[1][0] = cpx(cos(phi) / sqrt(2), sin(phi) / sqrt(2));
    g.mat[1][1] = cpx(cos(lambda + phi) / sqrt(2), sin(lambda + phi) / sqrt(2));
    g.name = "U2";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U3(int targetQubit, value_t theta, value_t phi, value_t lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U3;
    g.mat[0][0] = cpx(cos(theta / 2));
    g.mat[0][1] = cpx(-cos(lambda) * sin(theta / 2), -sin(lambda) * sin(theta / 2));
    g.mat[1][0] = cpx(cos(phi) * sin(theta / 2), sin(phi) * sin(theta / 2));
    g.mat[1][1] = cpx(cos(phi + lambda) * cos(theta / 2), sin(phi + lambda) * cos(theta / 2));
    g.name = "U3";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U(int targetQubit, std::vector<cpx> params) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U;
    g.mat[0][0] = params[0]; g.mat[0][1] = params[1];
    g.mat[1][0] = params[2]; g.mat[1][1] = params[3];
    g.name = "U";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::H(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::H;
    g.mat[0][0] = cpx(1/sqrt(2)); g.mat[0][1] = cpx(1/sqrt(2));
    g.mat[1][0] = cpx(1/sqrt(2)); g.mat[1][1] = cpx(-1/sqrt(2));
    g.name = "H";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::X(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::X;
    g.mat[0][0] = cpx(0); g.mat[0][1] = cpx(1);
    g.mat[1][0] = cpx(1); g.mat[1][1] = cpx(0);
    g.name = "X";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::Y(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::Y;
    g.mat[0][0] = cpx(0); g.mat[0][1] = cpx(0, -1);
    g.mat[1][0] = cpx(0, 1); g.mat[1][1] = cpx(0);
    g.name = "Y";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::Z(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::Z;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(-1);
    g.name = "Z";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::S(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::S;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(0, 1);
    g.name = "S";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::SDG(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::SDG;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(0, -1);
    g.name = "SDG";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::T(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::T;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(1/sqrt(2), 1/sqrt(2));
    g.name = "T";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::TDG(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::T;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(1/sqrt(2), -1/sqrt(2));
    g.name = "TDG";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RX(int targetQubit, value_t angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RX;
    g.mat[0][0] = cpx(cos(angle/2.0)); g.mat[0][1] = cpx(0, -sin(angle/2.0));
    g.mat[1][0] = cpx(0, -sin(angle/2.0)); g.mat[1][1] = cpx(cos(angle/2.0));
    g.name = "RX";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RY(int targetQubit, value_t angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RY;
    g.mat[0][0] = cpx(cos(angle/2.0)); g.mat[0][1] = cpx(-sin(angle/2.0));
    g.mat[1][0] = cpx(sin(angle/2.0)); g.mat[1][1] = cpx(cos(angle/2.0));
    g.name = "RY";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RZ(int targetQubit, value_t angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RZ;
    g.mat[0][0] = cpx(cos(angle/2), -sin(angle/2)); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(cos(angle/2), sin(angle/2));
    g.name = "RZ";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::ID(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::ID;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(1);
    g.name = "ID";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GII(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GII;
    g.mat[0][0] = cpx(0, 1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(0, 1);
    g.name = "GII";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GZZ(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GZZ;
    g.mat[0][0] = cpx(-1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(-1);
    g.name = "GZZ";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GOC(int targetQubit, value_t r, value_t i) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GOC;
    g.mat[0][0] = cpx(1); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(r, i);
    g.name = "GOC";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GCC(int targetQubit, value_t r, value_t i) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GCC;
    g.mat[0][0] = cpx(r, i); g.mat[0][1] = cpx(0);
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(r, i);
    g.name = "GCC";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RZZ(int targetQubit1, int targetQubit2, value_t theta) {
    // [00]  0    0    0
    //  0   [01]  0    0
    //  0    0   [01]  0
    //  0    0    0   [00]
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RZZ;
    g.mat[0][0] = cpx(cos(theta/2), -sin(theta/2)); g.mat[0][1] = cpx(cos(theta/2), sin(theta/2));
    g.mat[1][0] = cpx(0); g.mat[1][1] = cpx(0);
    g.name = "RZZ";
    g.encodeQubit = targetQubit1;
    g.targetQubit = targetQubit2;
    g.controlQubit = -3;
    return g;
}

Gate Gate::MCU(std::vector<int> controlQubits, int targetQubit, std::vector<cpx> params) {
    printf("[warning] MCU gate is not tested!\n");
    if (controlQubits.size() == 0) return Gate::U(targetQubit, params);
    if (controlQubits.size() == 1) return Gate::CU(controlQubits[0], targetQubit, params);
    Gate g;
    g.gateID = ++globalGateID;
    g.type = GateType::MCU;
    g.mat[0][0] = params[0]; g.mat[0][1] = params[1];
    g.mat[1][0] = params[2]; g.mat[1][1] = params[3];
    g.name = "MCU";
    g.encodeQubit = to_bitmap(controlQubits);
    g.targetQubit = targetQubit;
    g.controlQubit = -2;
    g.controlQubits = controlQubits;
    return g;
}

auto gen_01_float = []() {
    return rand() * 1.0 / RAND_MAX;
};
auto gen_0_2pi_float = []() {
        return gen_01_float() * acos(-1) * 2;
};

Gate Gate::random(int lo, int hi) {
    int type = rand() % int(GateType::TOTAL);
    return random(lo, hi, GateType(type));
}

Gate Gate::random(int lo, int hi, GateType type) {
    auto gen_c1_id = [lo, hi](int &t, int &c1) {
        assert(hi - lo >= 2);
        do {
            c1 = rand() % (hi - lo) + lo;
            t = rand() % (hi - lo) + lo;
        } while (c1 == t);
    };
    auto gen_single_id = [lo, hi](int &t) {
        t = rand() % (hi - lo) + lo;
    };
    switch (type) {
        case GateType::CNOT: {
            int t, c1;
            gen_c1_id(t, c1);
            return CNOT(c1, t);
        }
        case GateType::CY: {
            int t, c1;
            gen_c1_id(t, c1);
            return CY(c1, t);
        }
        case GateType::CZ: {
            int t, c1;
            gen_c1_id(t, c1);
            return CZ(c1, t);
        }
        case GateType::CRX: {
            int t, c1;
            gen_c1_id(t, c1);
            return CRX(c1, t, gen_0_2pi_float());
        }
        case GateType::CRY: {
            int t, c1;
            gen_c1_id(t, c1);
            return CRY(c1, t, gen_0_2pi_float());
        }
        case GateType::CU1: {
            int t, c1;
            gen_c1_id(t, c1);
            return CU1(c1, t, gen_0_2pi_float());
        }
        case GateType::CRZ: {
            int t, c1;
            gen_c1_id(t, c1);
            return CRZ(c1, t, gen_0_2pi_float());
        }
        case GateType::CU: {
            int t, c1;
            gen_c1_id(t, c1);
            std::vector<cpx> param = {
                cpx(gen_0_2pi_float(), gen_0_2pi_float()),
                cpx(gen_0_2pi_float(), gen_0_2pi_float()),
                cpx(gen_0_2pi_float(), gen_0_2pi_float()),
                cpx(gen_0_2pi_float(), gen_0_2pi_float())
            };
            return CU(c1, t, param);
        }
        case GateType::U1: {
            int t;
            gen_single_id(t);
            return U1(t, gen_0_2pi_float());
        }
        case GateType::U2: {
            int t;
            gen_single_id(t);
            return U2(t, gen_0_2pi_float(), gen_0_2pi_float());
        }
        case GateType::U3: {
            int t;
            gen_single_id(t);
            return U3(t, gen_0_2pi_float(), gen_0_2pi_float(), gen_0_2pi_float());
        }
        case GateType::U: {
            int t;
            gen_single_id(t);
            std::vector<cpx> param = {
                cpx(gen_0_2pi_float(), gen_0_2pi_float()),
                cpx(gen_0_2pi_float(), gen_0_2pi_float()),
                cpx(gen_0_2pi_float(), gen_0_2pi_float()),
                cpx(gen_0_2pi_float(), gen_0_2pi_float())
            };
            return U(t, param);
        }
        case GateType::H: {
            int t;
            gen_single_id(t);
            return H(t);
        }
        case GateType::X: {
            int t;
            gen_single_id(t);
            return X(t);
        }
        case GateType::Y: {
            int t;
            gen_single_id(t);
            return Y(t);
        }
        case GateType::Z: {
            int t;
            gen_single_id(t);
            return Z(t);
        }
        case GateType::S: {
            int t;
            gen_single_id(t);
            return S(t);
        }
        case GateType::SDG: {
            int t;
            gen_single_id(t);
            return SDG(t);
        }
        case GateType::T: {
            int t;
            gen_single_id(t);
            return T(t);
        }
        case GateType::TDG: {
            int t;
            gen_single_id(t);
            return TDG(t);
        }
        case GateType::RX: {
            int t;
            gen_single_id(t);
            return RX(t, gen_0_2pi_float());
        }
        case GateType::RY: {
            int t;
            gen_single_id(t);
            return RY(t, gen_0_2pi_float());
        }
        case GateType::RZ: {
            int t;
            gen_single_id(t);
            return RZ(t, gen_0_2pi_float());
        }
        default: {
            printf("invalid %d\n", (int) type);
            assert(false);
        }
    }
    exit(1);
}

Gate Gate::control(int controlQubit, int targetQubit, GateType type) {
    switch (type) {
        case GateType::CNOT: {
            return CNOT(controlQubit, targetQubit);
        }
        case GateType::CY: {
            return CY(controlQubit, targetQubit);
        }
        case GateType::CZ: {
            return CZ(controlQubit, targetQubit);
        }
        case GateType::CRX: {
            return CRX(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CRY: {
            return CRY(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CU1: {
            return CU1(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CRZ: {
            return CRZ(controlQubit, targetQubit, gen_0_2pi_float());
        }
        default: {
            assert(false);
        }
    }
    exit(1);
}

GateType Gate::toU(GateType type) {
    switch (type) {
        case GateType::CNOT:
            return GateType::X;
        case GateType::CY:
            return GateType::Y;
        case GateType::CZ:
            return GateType::Z;
        case GateType::CRX:
            return GateType::RX;
        case GateType::CRY:
            return GateType::RY;
        case GateType::CU1:
            return GateType::U1;
        case GateType::CRZ:
            return GateType::RZ;
        case GateType::CU:
            return GateType::U;
        default:
            UNREACHABLE()
    }
}

std::string Gate::get_name(GateType ty) {
    return random(0, 10, ty).name;
}

std::vector<unsigned char> Gate::serialize() const {
    auto name_len = name.length();
    int len =
        sizeof(name_len) + name.length() + 1 + sizeof(gateID) + sizeof(type) + sizeof(mat)
        + sizeof(targetQubit) + sizeof(controlQubit) + sizeof(encodeQubit);
    std::vector<unsigned char> ret; ret.resize(len);
    unsigned char* arr = ret.data();
    int cur = 0;
    SERIALIZE_STEP(gateID);
    SERIALIZE_STEP(type);
    memcpy(arr + cur, mat, sizeof(mat)); cur += sizeof(cpx) * 4;
    SERIALIZE_STEP(name_len);
    strcpy(reinterpret_cast<char*>(arr) + cur, name.c_str()); cur += name_len + 1;
    SERIALIZE_STEP(targetQubit);
    SERIALIZE_STEP(controlQubit);
    SERIALIZE_STEP(encodeQubit);
    assert(cur == len);
    return ret;
}

Gate Gate::deserialize(const unsigned char* arr, int& cur) {
    Gate g;
    DESERIALIZE_STEP(g.gateID);
    DESERIALIZE_STEP(g.type);
    memcpy(g.mat, arr + cur, sizeof(g.mat)); cur += sizeof(cpx) * 4;
    decltype(g.name.length()) name_len; DESERIALIZE_STEP(name_len);
    g.name = std::string(reinterpret_cast<const char*>(arr) + cur, name_len); cur += name_len + 1;
    DESERIALIZE_STEP(g.targetQubit);
    DESERIALIZE_STEP(g.controlQubit);
    DESERIALIZE_STEP(g.encodeQubit);
    return g;
}