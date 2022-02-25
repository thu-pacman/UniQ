#pragma once
#include "schedule.h"
#include "utils.h"
#include "gate.h"

#define GATE_NUM 24
#define MAX_QBITS 40

#define CALC_ALL_PARAM 0
#define CALC_PARTIAL_PARAM 1

/*
* build performance model to choose between BLAS and perGate backend
* Is a singleton class
**/
class Evaluator {
private:
    // pergate single gate performance for 512 runs with 28 qbits
    double pergate_single_perf[MAX_QBITS + 1][(size_t)GateType::TOTAL][LOCAL_QUBIT_SIZE];
    // pergate control gate performance for 512 runs with 28 qbits
    double pergate_ctr_perf[MAX_QBITS + 1][(size_t)GateType::TOTAL][LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE];
    // overhead of one pergate group
    double BLAS_perf[MAX_QBITS + 1][MAX_QBITS + 1];
    double cutt_cost[MAX_QBITS + 1];
    bool num_qbits_loaded_param[MAX_QBITS + 1];
    const double pergate_group_overhead = 1.0 / (1 << 27);

    int param_type;

    Evaluator();
    
    static Evaluator* instance_ptr;
public:
    static Evaluator* getInstance() {
        if(instance_ptr == nullptr) {
            instance_ptr = new Evaluator;
        }
        return instance_ptr;
    }
    void loadPergateSingle(int numQubits, FILE* qbit_param, GateType gate_type);
    void loadPergateCtr(int numQubits, FILE* qbit_param, GateType gate_type);
    void loadParam(int numQubits);
    double perfPerGate(int numQubits, const GateGroup* gg);
    double perfPerGate(int numQubits, const std::vector<GateType>& types);
    double perfBLAS(int numQubits, int blasSize);
    // return True if choose pergate over BLAS
    bool PerGateOrBLAS(const GateGroup* gg_pergate, const GateGroup* gg_blas, int numQubits, int blasSize);
};
