#include "schedule.h"
#include "logger.h"
#include "utils.h"
#include <cstring>
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <tuple>
#include <omp.h>
#include <dbg.h>

#ifdef USE_GPU
#include "cuda/entry.h"
#elif USE_CPU
#include "cpu/entry.h"
#endif

std::string to_string(Backend b) {
    switch (b) {
        case Backend::None: return "None";
        case Backend::PerGate: return "PerGate";
        case Backend::BLAS: return "BLAS";
    }
    UNREACHABLE();
}

GateGroup GateGroup::merge(const GateGroup& a, const GateGroup& b) {
    GateGroup ret;
    ret.relatedQubits = a.relatedQubits | b.relatedQubits;
    ret.gates = a.gates;
    std::vector<int> usedID;
    for (auto& g: a.gates) {
        usedID.push_back(g.gateID);
    }
    std::sort(usedID.begin(), usedID.end());
    for (auto& g: b.gates) {
        if (!std::binary_search(usedID.begin(), usedID.end(), g.gateID)) {
            ret.gates.push_back(g);
        }
    }
    return ret;
}

idx_t GateGroup::newRelated(idx_t relatedQubits, const Gate& gate, idx_t localQubits, bool enableGlobal) {
      if (enableGlobal) {
        if (!gate.isDiagonal()) {
            relatedQubits |= idx_t(1) << gate.targetQubit;
            if (gate.isTwoQubitGate()) {
                relatedQubits |= idx_t(1) << gate.encodeQubit;
            }
        }
    } else {
        if (!gate.isDiagonal() || (localQubits >> gate.targetQubit & 1))
            relatedQubits |= idx_t(1) << gate.targetQubit;
        if (gate.isControlGate() && (localQubits >> gate.controlQubit & 1))
            relatedQubits |= idx_t(1) << gate.controlQubit;
        if (gate.isMCGate())
            relatedQubits |= gate.encodeQubit;
        if (gate.isTwoQubitGate())
            relatedQubits |= idx_t(1) << gate.encodeQubit;
    }
    return relatedQubits;
 }

void GateGroup::addGate(const Gate& gate, idx_t localQubits, bool enableGlobal) {
    gates.push_back(gate);
    relatedQubits = newRelated(relatedQubits, gate, localQubits, enableGlobal);
}

GateGroup GateGroup::copyGates() {
    GateGroup ret;
    ret.gates = this->gates;
    ret.relatedQubits = this->relatedQubits;
    ret.backend = this->backend;
    return ret;
}

void Schedule::dump(int numQubits) {
    int L = 3;
    for (auto& lg: localGroups) {
        for (auto& gg: lg.overlapGroups) {
            switch (gg.backend) {
                case Backend::BLAS: printf("<BLAS>\n"); break;
                case Backend::PerGate: printf("<PerGate>\n"); break;
                case Backend::None: printf("<None\n>"); break;
            }
            for (const Gate& gate: gg.gates) {
                for (int i = 0; i < numQubits; i++) {
                    if (gate.hasControl(i)) {
                        printf(".");
                        for (int j = 1; j < L; j++) printf(" ");
                    } else if (gate.hasTarget(i)) {
                        printf("%s", gate.name.c_str());
                        for (int j = gate.name.length(); j < L; j++)
                            printf(" ");
                    } else {
                        if (gg.contains(i)) putchar('+');
                        else if (lg.contains(i)) putchar('/');
                        else putchar('|');
                        for (int j = 1; j < L; j++) printf(" ");
                    }
                }
                if (gate.controlErrors.size() > 0 || gate.targetErrors.size() > 0) {
                    printf("ERR: ");
                    if (gate.targetErrors.size() > 0) {
                        for (auto err: gate.targetErrors)
                            printf("%s ", Gate::get_name(err.type).c_str());
                    }
                    if (gate.controlErrors.size() > 0) {
                        printf("| ");
                        for (auto err: gate.controlErrors)
                            printf("%s ", Gate::get_name(err.type).c_str());
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
        for (int i = 0; i < numQubits * L; i++) {
            printf("-");
        }
        printf("\n");
        for (auto& gg: lg.fullGroups) {
            printf("%llx ", gg.relatedQubits);
            switch (gg.backend) {
                case Backend::BLAS: printf("<BLAS>\n"); break;
                case Backend::PerGate: printf("<PerGate>\n"); break;
                case Backend::None: printf("<None\n>"); break;
            }
            for (const Gate& gate: gg.gates) {
                for (int i = 0; i < numQubits; i++) {
                    if (gate.hasControl(i)) {
                        printf(".");
                        for (int j = 1; j < L; j++) printf(" ");
                    } else if (gate.hasTarget(i)) {
                        printf("%s", gate.name.c_str());
                        for (int j = gate.name.length(); j < L; j++)
                            printf(" ");
                    } else {
                        if (gg.contains(i)) putchar('+');
                        else if (lg.contains(i)) putchar('/');
                        else putchar('|');
                        for (int j = 1; j < L; j++) printf(" ");
                    }
                }
                if (gate.controlErrors.size() > 0 || gate.targetErrors.size() > 0) {
                    printf("ERR: ");
                    if (gate.targetErrors.size() > 0) {
                        for (auto err: gate.targetErrors)
                            printf("%s ", Gate::get_name(err.type).c_str());
                    }
                    if (gate.controlErrors.size() > 0) {
                        printf("| ");
                        for (auto err: gate.controlErrors)
                            printf("%s ", Gate::get_name(err.type).c_str());
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
        for (int i = 0; i < numQubits * L; i++) {
            printf("#");
        }
        printf("\n\n");
    }
    fflush(stdout);
#if GPU_BACKEND == 1 || GPU_BACKEND == 2 || GPU_BACKEND == 3 || GPU_BACKEND == 4 || GPU_BACKEND == 5
    for (size_t i = 0; i < localGroups.size(); i++) {
        const LocalGroup& lg = localGroups[i];
        printf("Global: ");
        for (int j = 0; j < numQubits; j++) {
            if (!(lg.relatedQubits >> j & 1)) {
                printf("%d ", j);
            }
        }
        printf("\n");
        printf("pos: ");
        for (int j = 0; j < numQubits; j++) {
            printf("[%d: %d] ", j, lg.state.pos[j]);
        }
        printf("\n");
        printf("layout: ");
        for (int j = 0; j < numQubits; j++) {
            printf("[%d: %d] ", j, lg.state.pos[j]);
        }
        printf("\n\n");
    }
#endif
    fflush(stdout);
}

std::vector<unsigned char> State::serialize() const {
    assert(pos.size() == layout.size());
    auto num_ele = pos.size();

    std::vector<unsigned char> result;
    result.resize(sizeof(num_ele));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(num_ele);
    SERIALIZE_VECTOR(pos, result);
    SERIALIZE_VECTOR(layout, result);
    return result;
}

State State::deserialize(const unsigned char* arr, int& cur) {
    State s;
    decltype(s.pos.size()) num_ele;

    DESERIALIZE_STEP(num_ele);
    DESERIALIZE_VECTOR(s.pos, num_ele);
    DESERIALIZE_VECTOR(s.layout, num_ele);

    return s;
}

std::vector<unsigned char> GateGroup::serialize() const {
    auto num_gates = gates.size();
    auto num_perm = cuttPerm.size();

    std::vector<unsigned char> result;
    result.resize(sizeof(num_gates) + sizeof(relatedQubits) + sizeof(num_perm) + sizeof(matQubit) + sizeof(Backend));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(num_gates);
    SERIALIZE_STEP(relatedQubits);
    SERIALIZE_STEP(num_perm);
    SERIALIZE_STEP(matQubit);
    SERIALIZE_STEP(backend);

    auto s = state.serialize();
    result.insert(result.end(), s.begin(), s.end());

    for (auto& gate: gates) {
        auto g = gate.serialize();
        result.insert(result.end(), g.begin(), g.end());
    }

    SERIALIZE_VECTOR(cuttPerm, result);

    return result;
}

GateGroup GateGroup::deserialize(const unsigned char* arr, int& cur) {
    GateGroup gg;
    decltype(gg.gates.size()) num_gates;
    decltype(gg.cuttPerm.size()) num_perm;

    DESERIALIZE_STEP(num_gates);
    DESERIALIZE_STEP(gg.relatedQubits);
    DESERIALIZE_STEP(num_perm);
    DESERIALIZE_STEP(gg.matQubit);
    DESERIALIZE_STEP(gg.backend);

    gg.state = State::deserialize(arr, cur);

    for (decltype(num_gates) i = 0; i < num_gates; i++) {
        gg.gates.push_back(Gate::deserialize(arr, cur));
    }

    DESERIALIZE_VECTOR(gg.cuttPerm, num_perm);

    return gg;
}

std::vector<unsigned char> LocalGroup::serialize() const {
    auto num_a2a = a2aComm.size();
    auto num_perm = cuttPerm.size();
    auto num_og = overlapGroups.size();
    auto num_fg = fullGroups.size();

    std::vector<unsigned char> result;
    result.resize(sizeof(a2aCommSize) + sizeof(num_a2a) + sizeof(num_perm) + sizeof(num_og) + sizeof(num_fg) + sizeof(relatedQubits));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(a2aCommSize);
    SERIALIZE_STEP(num_a2a);
    SERIALIZE_STEP(num_perm);
    SERIALIZE_STEP(num_og);
    SERIALIZE_STEP(num_fg);
    SERIALIZE_STEP(relatedQubits);
    
    SERIALIZE_VECTOR(a2aComm, result);
    SERIALIZE_VECTOR(cuttPerm, result);

    auto s = state.serialize();
    result.insert(result.end(), s.begin(), s.end());

    for (auto& gateGroup: overlapGroups) {
        auto gg = gateGroup.serialize();
        result.insert(result.end(), gg.begin(), gg.end());
    }

    for (auto& gateGroup: fullGroups) {
        auto gg = gateGroup.serialize();
        result.insert(result.end(), gg.begin(), gg.end());
    }

    return result;
}


LocalGroup LocalGroup::deserialize(const unsigned char* arr, int& cur) {
    LocalGroup s;
    decltype(s.a2aComm.size()) num_a2a;
    decltype(s.cuttPerm.size()) num_perm;
    decltype(s.overlapGroups.size()) num_og;
    decltype(s.fullGroups.size()) num_fg;

    DESERIALIZE_STEP(s.a2aCommSize);
    DESERIALIZE_STEP(num_a2a);
    DESERIALIZE_STEP(num_perm);
    DESERIALIZE_STEP(num_og);
    DESERIALIZE_STEP(num_fg);
    DESERIALIZE_STEP(s.relatedQubits);

    DESERIALIZE_VECTOR(s.a2aComm, num_a2a);
    DESERIALIZE_VECTOR(s.cuttPerm, num_perm);

    s.state = State::deserialize(arr, cur);

    for (decltype(num_og) i = 0; i < num_og; i++) {
        s.overlapGroups.push_back(GateGroup::deserialize(arr, cur));
    }

    for (decltype(num_fg) i = 0; i < num_fg; i++) {
        s.fullGroups.push_back(GateGroup::deserialize(arr, cur));
    }

    return s;
}

std::vector<unsigned char> Schedule::serialize() const {
    auto num_lg = localGroups.size();
    std::vector<unsigned char> result;
    result.resize(sizeof(num_lg));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(num_lg);

    auto s = finalState.serialize();
    result.insert(result.end(), s.begin(), s.end());

    for (auto& localGroup: localGroups) {
        auto lg = localGroup.serialize();
        result.insert(result.end(), lg.begin(), lg.end());
    }
    return result;
}


Schedule Schedule::deserialize(const unsigned char* arr, int& cur) {
    Schedule s;
    decltype(s.localGroups.size()) num_lg;
    DESERIALIZE_STEP(num_lg);
    s.finalState = State::deserialize(arr, cur);
    for (decltype(num_lg) i = 0; i < num_lg; i++) {
        s.localGroups.push_back(LocalGroup::deserialize(arr, cur));
    }
    return s;
}

std::vector<int> gen_perm_vector(int len) {
    std::vector<int> ret;
    for (int i = 0; i < len; i++)
        ret.push_back(i);
    return ret;
}

State GateGroup::initPerGateState(const State& oldState) {
    state = oldState;
    return state;
}

State GateGroup::initBlasState(const State& oldState, int numLocalQubits) {
    std::vector<int> pos = oldState.pos;
    std::vector<int> layout = oldState.layout;

    std::vector<int> toGlobal; // qubit id
    std::vector<int> toLocal; // qubit id
    int numMatQubits = bitCount(relatedQubits);
    for (int i = 0; i < numMatQubits; i++) {
        int q = layout[i];
        if (!(relatedQubits >> q & 1))
            toGlobal.push_back(q);
    }
    for (int i = numMatQubits; i < numLocalQubits; i++) {
        int q = layout[i];
        if (relatedQubits >> q & 1)
            toLocal.push_back(q);
    }
    assert(toGlobal.size() == toLocal.size());
    cuttPerm = gen_perm_vector(numLocalQubits);
    for (size_t i = 0; i < toGlobal.size(); i++) {
        int x = toGlobal[i], y = toLocal[i];
        std::swap(cuttPerm[pos[x]], cuttPerm[pos[y]]);
        std::swap(pos[x], pos[y]);
        layout[pos[x]] = x; layout[pos[y]] = y;
    }

    
#ifdef SHOW_SCHEDULE
    printf("perm: "); for (auto x: cuttPerm) printf("%d ", x); printf("\n");
    printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
    printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n\n");
#endif
    // complex have two floats -> use double2
    // cuttPerm.push_back(0);
    // for (int i = cuttPerm.size() - 1; i; i--) {
    //     cuttPerm[i] = cuttPerm[i-1] + 1;
    // }
    // cuttPerm[0] = 0;
    
    this->matQubit = std::max(numMatQubits, MIN_MAT_SIZE);
    State newState = State(pos, layout);
    this->state = newState;
    return newState;
}

void GateGroup::getCuttPlanPointers(int numLocalQubits, std::vector<transHandle*> &transPlanPointers, std::vector<int*> &transPermPointers, std::vector<int> &locals) {
    transPlans.clear();
    if (backend != Backend::BLAS)
        return;
    int startSize = transPlans.size();
    transPlans.resize(startSize + MyGlobalVars::localGPUs);
    transPlanPointers.push_back(transPlans.data() + startSize);
    transPermPointers.push_back(cuttPerm.data());
    locals.push_back(numLocalQubits);
}

State GateGroup::initState(const State& oldState, int numLocalQubits) {
    if (backend == Backend::PerGate) {
        return initPerGateState(oldState);
    }
    if (backend == Backend::BLAS) {
        return initBlasState(oldState, numLocalQubits);
    }
    UNREACHABLE();
}

State LocalGroup::initState(const State& oldState, int numQubits, const std::vector<int>& newGlobals, idx_t overlapGlobals, idx_t overlapRelated, int globalBit) {
    int numLocalQubits = numQubits - globalBit;
    auto pos = oldState.pos, layout = oldState.layout;
    int overlapCnt = bitCount(overlapGlobals);
    cuttPerm = gen_perm_vector(numLocalQubits);
    std::vector<int> newBuffer;
    if (overlapCnt > 0) {
        int need = overlapCnt;
        for (int i = numLocalQubits - 1; i >= 0; i--) {
            int x = layout[i];
            if (std::find(newGlobals.begin(), newGlobals.end(), x) == newGlobals.end() && !(overlapRelated >> x & 1)) {
                newBuffer.push_back(x);
                need --;
                if (need == 0)
                    break;
            }
        }
    }
    for (int i = 0; i < globalBit; i++)
        if (!(overlapGlobals >> i & 1))
            newBuffer.push_back(newGlobals[i]);
    assert(int(newBuffer.size()) == globalBit);
    for (int i = 0, c = numLocalQubits - globalBit; i < globalBit; i++, c++) {
        if (layout[c] == newBuffer[i])
            continue;
        std::swap(cuttPerm[pos[newBuffer[i]]], cuttPerm[c]);
        int swappedQid = layout[c];
        pos[swappedQid] = pos[newBuffer[i]];
        pos[newBuffer[i]] = c;
        layout[pos[newBuffer[i]]] = newBuffer[i];
        layout[pos[swappedQid]] = swappedQid;
    }
#ifdef SHOW_SCHEDULE
    printf("buffer: "); for (auto x: newBuffer) printf("%d ", x); printf("\n");
    printf("perm: "); for (auto x: cuttPerm) printf("%d ", x); printf("\n");
    printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
    printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n\n");
#endif
    // complex have two floats -> use double2
    // cuttPerm.push_back(0);
    // for (int i = cuttPerm.size() - 1; i; i--) {
    //     cuttPerm[i] = cuttPerm[i-1] + 1;
    // }
    // cuttPerm[0] = 0;

    int c = numLocalQubits - globalBit + overlapCnt;
    for (int i = 0; i < globalBit; i++) {
        if (overlapGlobals >> i & 1) continue;
        int a = i + numLocalQubits;
        int qa = layout[a], qc = layout[c];
        layout[a] = qc; pos[qc] = a;
        layout[c] = qa; pos[qa] = c;
        c++;
    }
#ifdef SHOW_SCHEDULE
    printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
    printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n");
    printf("------------------------------------------------------\n");
#endif
    std::vector<std::pair<int, int>> newCommPair;
#if MODE == 2
#ifdef SHOW_SCHEDULE
    if (overlapGlobals > 0) printf("[warning] overlap global = %x, not verified\n", (int) overlapGlobals);
#endif
    overlapGlobals = duplicate_bit(overlapGlobals);
#endif
    for (int i = 0; i < MyGlobalVars::numGPUs; i++) {
        newCommPair.push_back(std::make_pair(i & overlapGlobals, i));
    }
    std::sort(newCommPair.begin(), newCommPair.end());
    std::vector<int> newComm;
    for (auto x: newCommPair) {
        newComm.push_back(x.second);
    }
    a2aComm = newComm;
    a2aCommSize = MyGlobalVars::numGPUs >> bitCount(overlapGlobals);
    auto newState = State(pos, layout);
    this->state = newState;
    return newState;
}

State LocalGroup::initStateInplace(const State& oldState, int numQubits, const std::vector<int>& newGlobals, idx_t overlapGlobals, int globalBit) {
    int localBit = MODE == 2 ? MyGlobalVars::bit / 2 : MyGlobalVars::bit;
    int numLocalQubits = numQubits - localBit;
    auto pos = oldState.pos, layout = oldState.layout;
#if MODE == 2
    overlapGlobals = duplicate_bit(overlapGlobals);
#endif
    int overlapCnt = bitCount(overlapGlobals);
    std::vector<int> oldGlobals;
    for (int i = 0; i < localBit; i++) {
        oldGlobals.push_back(layout[i + numLocalQubits]);
    }
    assert(int(oldGlobals.size()) == localBit);
    std::vector<int> newPos;
    for (size_t i = 0; i < oldGlobals.size(); i++) {
        if (oldGlobals[i] != newGlobals[i])
            newPos.push_back(pos[newGlobals[i]]);
    }
    std::sort(newPos.begin(), newPos.end());
    int psCnt = 0;
    for (size_t i = 0; i < oldGlobals.size(); i++) {
        if (oldGlobals[i] != newGlobals[i]) {
            int x = oldGlobals[i];
            int y = layout[newPos[psCnt]];
            layout[pos[x]] = y; layout[pos[y]] = x;
            std::swap(pos[x], pos[y]);
            psCnt++;
        }
    }
#ifdef SHOW_SCHEDULE
    printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
    printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n");
    printf("------------------------------------------------------\n");
#endif
    auto newState = State(pos, layout);
    this->state = newState;
    std::vector<std::pair<int, int>> newCommPair;
    for (int i = 0; i < MyGlobalVars::numGPUs; i++) {
        newCommPair.push_back(std::make_pair(i & overlapGlobals, i));
    }
    std::sort(newCommPair.begin(), newCommPair.end());
    std::vector<int> newComm;
    for (auto x: newCommPair) {
        newComm.push_back(x.second);
    }
    a2aComm = newComm;
    a2aCommSize = 1 << (MyGlobalVars::bit - overlapCnt);
    return newState;
}

State LocalGroup::initFirstGroupState(const State& oldState, int numQubits, const std::vector<int>& newGlobals) {
#if MODE == 2
    int numLocalQubits = numQubits - MyGlobalVars::bit / 2;
#else
    int numLocalQubits = numQubits - MyGlobalVars::bit;
#endif
    auto pos = oldState.pos, layout = oldState.layout;
    assert(overlapGroups.size() == 0);
    for (size_t i = 0; i < newGlobals.size(); i++) {
        int x = newGlobals[i];
        if (pos[x] >= numLocalQubits)
            continue;
        for (int y = numLocalQubits; y < numQubits; y++) {
            if (std::find(newGlobals.begin(), newGlobals.end(), layout[y]) == newGlobals.end()) {
                std::swap(pos[x], pos[y]);
                layout[pos[x]] = x; layout[pos[y]] = y;
                break;
            }
        }
        
    }
    state = State(pos, layout);
    a2aCommSize = -1;
    a2aComm.clear();
#ifdef SHOW_SCHEDULE
    printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
    printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n");
    printf("------------------------------------------------------\n");
#endif
    auto newState = State(pos, layout);
    this->state = newState;
    return newState;
}

void LocalGroup::getCuttPlanPointers(int numLocalQubits, std::vector<transHandle*> &transPlanPointers, std::vector<int*> &transPermPointers, std::vector<int> &locals, bool isFirstGroup) {
    transPlans.clear();
#if not INPLACE > 0
    if (isFirstGroup) {
        for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
            transPlans.push_back(transHandle());
        }
    } else {
        int startSize = transPlans.size();
        transPlans.resize(startSize + MyGlobalVars::localGPUs);
        transPlanPointers.push_back(transPlans.data() + startSize);
        transPermPointers.push_back(cuttPerm.data());
        locals.push_back(numLocalQubits);
    }
#endif
    for (auto& gg: overlapGroups) {
        gg.getCuttPlanPointers(numLocalQubits - MyGlobalVars::bit, transPlanPointers, transPermPointers, locals);
    }
    for (auto& gg: fullGroups) {
        gg.getCuttPlanPointers(numLocalQubits, transPlanPointers, transPermPointers, locals);
    }
}

#define APPLY_IDENTITY_GATE(val) \
for (int i = 0; i < n * n; i++) { \
    mat[i] = mat[i] * val; \
}

#define APPLY_CI_GATE(val) \
for (int i = 0; i < n; i++) { \
    for (int j = 0; j < (n >> 1); j++) { \
        int lo = j; \
        lo = insertBit(lo, c1); \
        lo += i * n; \
        int hi = lo | 1 << c1; \
        mat[hi] = mat[hi] * val; \
    } \
}

#define APPLY_SINGLE_GATE() \
for (int i = 0; i < n; i++) { \
    for (int j = 0; j < (n >> 1); j++) { \
        int lo = j; \
        lo = insertBit(lo, t); \
        lo += i * n; \
        int hi = lo | 1 << t; \
        cpx v0 = mat[lo]; \
        cpx v1 = mat[hi]; \
        mat[lo] = v0 * cpx(gate.mat[0][0]) + v1 * cpx(gate.mat[0][1]); \
        mat[hi] = v0 * cpx(gate.mat[1][0]) + v1 * cpx(gate.mat[1][1]); \
    } \
}

#define APPLY_CONTROL_GATE() \
for (int i = 0; i < n; i++) { \
    for (int j = 0; j < (n >> 2); j++) { \
        int lo = j; \
        lo = insertBit(lo, b); \
        lo = insertBit(lo, a); \
        lo += i * n; \
        lo |= 1 << c1; \
        int hi = lo | 1 << t; \
        cpx v0 = mat[lo]; \
        cpx v1 = mat[hi]; \
        mat[lo] = v0 * cpx(gate.mat[0][0]) + v1 * cpx(gate.mat[0][1]); \
        mat[hi] = v0 * cpx(gate.mat[1][0]) + v1 * cpx(gate.mat[1][1]); \
    } \
}

#define APPLY_MUU_GATE() \
for (int i = 0; i < n; i++) { \
    for (int j = 0; j < (n >> 2); j++) { \
        int s00 = j; \
        s00 = insertBit(s00, b); \
        s00 = insertBit(s00, a); \
        s00 += i * n; \
        int s01 = s00 | (1 << t1); \
        int s10 = s00 | (1 << t2); \
        int s11 = s01 | s10; \
        cpx v00 = mat[s00]; \
        cpx v01 = mat[s01]; \
        cpx v10 = mat[s10]; \
        cpx v11 = mat[s11]; \
        mat[s00] = v00 * cpx(gate.mat[0][0]) + v11 * cpx(gate.mat[1][1]); \
        mat[s01] = v01 * cpx(gate.mat[0][1]) + v10 * cpx(gate.mat[1][0]); \
        mat[s10] = v01 * cpx(gate.mat[1][0]) + v10 * cpx(gate.mat[0][1]); \
        mat[s11] = v00 * cpx(gate.mat[1][1]) + v11 * cpx(gate.mat[0][0]); \
    } \
}

#define APPLY_MC_GATE() \
for (int i = 0; i < n; i++) { \
    for (int j = 0; j < (n >> 1); j++) { \
        int lo = j; \
        lo = insertBit(lo, t); \
        if ((lo & cbits) != cbits) continue; \
        lo += i * n; \
        int hi = lo | 1 << t; \
        cpx v0 = mat[lo]; \
        cpx v1 = mat[hi]; \
        mat[lo] = v0 * cpx(gate.mat[0][0]) + v1 * cpx(gate.mat[0][1]); \
        mat[hi] = v0 * cpx(gate.mat[1][0]) + v1 * cpx(gate.mat[1][1]); \
    } \
}

#define APPLY_MC_IDENTITY_GATE(val) \
for (int i = 0; i < n * n; i++) { \
    if ((i & cbits) == cbits) \
        mat[i] = mat[i] * val; \
}


void GateGroup::initCPUMatrix(int numLocalQubit) {
    auto& pos = state.pos;
    int numMatQubits = this->matQubit;
    assert(numMatQubits <= std::max(BLAS_MAT_LIMIT, MIN_MAT_SIZE));
    int n = 1 << numMatQubits;
    matrix.clear();
    matrix.resize(MyGlobalVars::localGPUs);
    for (int gpuID = 0; gpuID < MyGlobalVars::localGPUs; gpuID++) {
        matrix[gpuID] = std::make_unique<cpx[]>(n * n);
    }
    #pragma omp parallel
    for (int gpuID = 0; gpuID < MyGlobalVars::localGPUs; gpuID++) {
        cpx* mat = matrix[gpuID].get();
        int globalGPUID = MyMPI::rank * MyGlobalVars::localGPUs + gpuID;
        auto isHiGPU = [globalGPUID, numLocalQubit](int q) {
            assert(q >= numLocalQubit);
            assert((q - numLocalQubit) < MyGlobalVars::bit);
            return (bool)(globalGPUID >> (q - numLocalQubit) & 1);
        };
        #pragma omp for
        for (int i = 0; i < n * n; i++) mat[i] = cpx(0.0, 0.0);

        #pragma omp for
        for (int i = 0; i < n; i++) {
            mat[i * n + i] = cpx(1.0, 0.0);
        }

        auto insertBit = [](int x, int pos) {
            return (x >> pos << (pos + 1)) | (x & ((idx_t(1) << pos) - 1));
        };

        for (auto& gate: gates) {
            if (gate.isMCGate()) {
                int t = pos[gate.targetQubit];
                idx_t cbits = 0;
                for (auto q: gate.controlQubits) {
                    int c = pos[q];
                    if (c >= numLocalQubit && !isHiGPU(c))
                        continue;
                    assert(c < numMatQubits);
                    cbits |= 1ll << c;
                }
                if (t >= numLocalQubit) {
                    bool isHi = isHiGPU(t);
                    auto val = gate.mat[isHi][isHi];
                    #pragma omp for
                    APPLY_MC_IDENTITY_GATE(val);
                } else {
                    #pragma omp for
                    APPLY_MC_GATE();
                }
            } else if (gate.isTwoQubitGate()) {
                int t1 = pos[gate.encodeQubit];
                int t2 = pos[gate.targetQubit];
                assert(t1 < numMatQubits || (t1 >= numLocalQubit && gate.isDiagonal()));
                assert(t2 < numMatQubits || (t2 >= numLocalQubit && gate.isDiagonal()));
                // sort
                int a = std::max(t1, t2);
                int b = std::min(t1, t2);
                if (t1 >= numLocalQubit || t2 >= numLocalQubit) {
                    UNIMPLEMENTED();
                }
                #pragma omp for
                APPLY_MUU_GATE()
            } else if (gate.isControlGate()) {
                int c1 = pos[gate.controlQubit];
                int t = pos[gate.targetQubit];
                assert(c1 < numMatQubits || c1 >= numLocalQubit);
                assert(t < numMatQubits || (t >= numLocalQubit && gate.isDiagonal()));
                // sort
                int a = std::max(c1, t);
                int b = std::min(c1, t);
                if (c1 >= numLocalQubit) {
                    if (!isHiGPU(c1)) continue;
                    if (t >= numLocalQubit) {
                        bool isHi = isHiGPU(t);
                        auto val = gate.mat[isHi][isHi];
                        #pragma omp for
                        APPLY_IDENTITY_GATE(val);
                    } else {
                        #pragma omp for
                        APPLY_SINGLE_GATE()
                    }
                } else {
                    if (t >= numLocalQubit) {
                        bool isHi = isHiGPU(t);
                        auto val = gate.mat[isHi][isHi];
                        #pragma omp for
                        APPLY_CI_GATE(val);
                    } else {
                        #pragma omp for
                        APPLY_CONTROL_GATE()
                    }
                }
            } else {
                int t = pos[gate.targetQubit];
                assert(t < numMatQubits || (t >= numLocalQubit && gate.isDiagonal()));
                if (t >= numLocalQubit) {
                    bool hi = isHiGPU(t);
                    #pragma omp for
                    APPLY_IDENTITY_GATE(gate.mat[hi][hi]);
                } else {
                    #pragma omp for
                    APPLY_SINGLE_GATE()
                }
            }
        }
        // assert(isUnitary(mat, n));
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++)
        //         printf("(%.2f %.2f) ", mat[i * n + j].x, mat[i * n + j].y);
        //     printf("\n");
        // }
    }
}

void GateGroup::initGPUMatrix() {
#ifdef USE_GPU
    CudaImpl::initGPUMatrix(deviceMats, matQubit, matrix);
#else
    UNIMPLEMENTED();
#endif
}

void GateGroup::initMatrix(int numLocalQubit) {
    if (backend == Backend::BLAS) {
        initCPUMatrix(numLocalQubit);
        initGPUMatrix();
    } else {
        UNREACHABLE();
    }
}


#if GPU_BACKEND == 1 || GPU_BACKEND == 2 || GPU_BACKEND == 3 || GPU_BACKEND == 4 || GPU_BACKEND == 5
void Schedule::initMatrix(int numQubits) {
    for (auto& lg: localGroups) {
        for (auto& gg: lg.overlapGroups) {
            if (gg.backend == Backend::BLAS)
                gg.initMatrix(numQubits - 2 * MyGlobalVars::bit);
        }
        for (auto& gg: lg.fullGroups) {
            if (gg.backend == Backend::BLAS)
                gg.initMatrix(numQubits - MyGlobalVars::bit);
        }
    }
}

#else
void Schedule::initMatrix(int numQubits) {
    UNREACHABLE()
}
#endif


void Schedule::initCuttPlans(int numLocalQubits) {
    std::vector<transHandle*> transPlanPointers;
    std::vector<int*> transPermPointers;
    std::vector<int> locals;
    for (size_t i = 0; i < localGroups.size(); i++) {
        localGroups[i].getCuttPlanPointers(numLocalQubits, transPlanPointers, transPermPointers, locals, i == 0);
    }

    assert(transPlanPointers.size() == transPermPointers.size());
    assert(transPermPointers.size() == locals.size());

#ifdef USE_GPU
    CudaImpl::initCuttPlans(transPlanPointers, transPermPointers, locals, numLocalQubits);
#else
    CpuImpl::initHpttPlans(transPlanPointers, transPermPointers, locals, numLocalQubits);
#endif
}

void removeGates(std::vector<Gate>& remain, const std::vector<Gate>& remove) {
    std::vector<int> usedID;
    for (auto& g: remove) usedID.push_back(g.gateID);
    std::sort(usedID.begin(), usedID.end());
    auto temp = remain;
    remain.clear();
    for (auto& g: temp) {
        if (!std::binary_search(usedID.begin(), usedID.end(), g.gateID)) {
            remain.push_back(g);
        }
    }
}
