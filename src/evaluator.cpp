#include "evaluator.h"
#include <cstring>

Evaluator* Evaluator::instance_ptr = nullptr;

Evaluator::Evaluator() {
    memset(num_qbits_loaded_param, 0, sizeof(num_qbits_loaded_param));
}

void Evaluator::loadPergateSingle(int numQubits, FILE* qbit_param, GateType gate_type) {
    if(param_type == CALC_ALL_PARAM) {
        for(int i = 0; i < LOCAL_QUBIT_SIZE; i++) {
            fscanf(qbit_param, "%lf", &pergate_single_perf[numQubits][int(gate_type)][i]);
        }
    }
    else {
        fscanf(qbit_param, "%lf", &pergate_single_perf[numQubits][int(gate_type)][1]);
    }
}

void Evaluator::loadPergateCtr(int numQubits, FILE* qbit_param, GateType gate_type) {
    if(param_type == CALC_ALL_PARAM) {
        for(int i = 0; i < LOCAL_QUBIT_SIZE; i++)
            for(int j = 0; j < LOCAL_QUBIT_SIZE; j++) {
                fscanf(qbit_param, "%lf", &pergate_ctr_perf[numQubits][int(gate_type)][i][j]);
            }
    }
    else {
        fscanf(qbit_param, "%lf", &pergate_ctr_perf[numQubits][int(gate_type)][0][2]);       
    }
} 

void Evaluator::loadParam(int numQubits) {
    if(num_qbits_loaded_param[numQubits])
        return;
#ifdef USE_EVALUATOR_PREPROCESS    
    FILE* qbit_param;
    std::string param_file_name = std::string("../evaluator-preprocess/parameter-files/") 
        + std::to_string(numQubits) + std::string("qubits.out");
    if((qbit_param = fopen(param_file_name.c_str(), "r"))) {
        fscanf(qbit_param, "%d", &param_type);

        loadPergateSingle(numQubits, qbit_param, GateType::U1);
        loadPergateSingle(numQubits, qbit_param, GateType::U2);
        loadPergateSingle(numQubits, qbit_param, GateType::U3);
        loadPergateSingle(numQubits, qbit_param, GateType::U);
        loadPergateSingle(numQubits, qbit_param, GateType::H );
        loadPergateSingle(numQubits, qbit_param, GateType::X );
        loadPergateSingle(numQubits, qbit_param, GateType::Y );
        loadPergateSingle(numQubits, qbit_param, GateType::Z );
        loadPergateSingle(numQubits, qbit_param, GateType::S );
        loadPergateSingle(numQubits, qbit_param, GateType::SDG);
        loadPergateSingle(numQubits, qbit_param, GateType::T );
        loadPergateSingle(numQubits, qbit_param, GateType::TDG);
        loadPergateSingle(numQubits, qbit_param, GateType::RX);
        loadPergateSingle(numQubits, qbit_param, GateType::RY);
        loadPergateSingle(numQubits, qbit_param, GateType::RZ);

        loadPergateCtr(numQubits, qbit_param, GateType::CNOT);
        loadPergateCtr(numQubits, qbit_param, GateType::CY  );
        loadPergateCtr(numQubits, qbit_param, GateType::CZ  );
        loadPergateCtr(numQubits, qbit_param, GateType::CRX );
        loadPergateCtr(numQubits, qbit_param, GateType::CRY );
        loadPergateCtr(numQubits, qbit_param, GateType::CU1 );
        loadPergateCtr(numQubits, qbit_param, GateType::CRZ );
        loadPergateCtr(numQubits, qbit_param, GateType::CU );

        for (int K = 1, i = 0; K < 1024; K <<= 1, i++) {
            fscanf(qbit_param, "%*d%lf", &BLAS_perf[numQubits][i]);
        }
        fscanf(qbit_param, "%lf", &cutt_cost[numQubits]);
        fclose(qbit_param);
    } else {
        printf("Parameter file not find for qubit number %d\n", numQubits);
        fflush(stdout);
        exit(1);
    }
    num_qbits_loaded_param[numQubits] = true;
#else
    printf("Use option USE_EVALUATOR_PREPROCESS for non-default qubit number %d\n", numQubits);
    fflush(stdout);
    exit(1);
#endif
}

double Evaluator::perfPerGate(int numQubits, const GateGroup* gg) {
    double tim_pred = 0;
    loadParam(numQubits);
    for(auto gate : (gg -> gates)) {
        switch(gate.type) {
            case GateType::CNOT : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CNOT)][0][2]; break;
            case GateType::CY : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CY)][0][2]; break;
            case GateType::CZ : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CZ)][0][2]; break;
            case GateType::CRX : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CRX)][0][2]; break;
            case GateType::CRY : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CRY)][0][2]; break;
            case GateType::CU1 :
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CU1)][0][2]; break;
            case GateType::CRZ : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CRZ)][0][2]; break;
            case GateType::CU : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CU)][0][2]; break;
            case GateType::U1 : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U1)][1]; break;
            case GateType::U2 : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U2)][1]; break;
            case GateType::U3 : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U3)][1]; break;
            case GateType::U : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U)][1]; break;
            case GateType::H : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::H)][1]; break;
            case GateType::X : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::X)][1]; break;
            case GateType::Y : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::Y)][1]; break;
            case GateType::Z : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::Z)][1]; break;
            case GateType::S : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::S)][1]; break;
            case GateType::SDG : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::SDG)][1]; break;
            case GateType::T : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::T)][1]; break;
            case GateType::TDG : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::TDG)][1]; break;
            case GateType::RX : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::RX)][1]; break;
            case GateType::RY : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::RY)][1]; break;
            case GateType::RZ : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::RZ)][1]; break;
            default:
                printf("meet wrong gate : %s\n", Gate::get_name(gate.type).c_str());
                UNREACHABLE()
        }
    }
    return tim_pred / 1000 / 512 + pergate_group_overhead * (1 << numQubits);
}

double Evaluator::perfPerGate(int numQubits, const std::vector<GateType>& types) {
    double tim_pred = 0;
    loadParam(numQubits);
    for(auto ty : types) {
        switch(ty) {
            case GateType::CNOT : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CNOT)][0][2]; break;
            case GateType::CY : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CY)][0][2]; break;
            case GateType::CZ : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CZ)][0][2]; break;
            case GateType::CRX : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CRX)][0][2]; break;
            case GateType::CRY : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CRY)][0][2]; break;
            case GateType::CU1 :
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CU1)][0][2]; break;
            case GateType::CRZ : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CRZ)][0][2]; break;
            case GateType::CU : 
                tim_pred += pergate_ctr_perf[numQubits][int(GateType::CU)][0][2]; break;
            case GateType::U1 : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U1)][1]; break;
            case GateType::U2 : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U2)][1]; break;
            case GateType::U3 : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U3)][1]; break;
            case GateType::U : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::U)][1]; break;
            case GateType::H : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::H)][1]; break;
            case GateType::X : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::X)][1]; break;
            case GateType::Y : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::Y)][1]; break;
            case GateType::Z : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::Z)][1]; break;
            case GateType::S : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::S)][1]; break;
            case GateType::SDG : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::SDG)][1]; break;
            case GateType::T : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::T)][1]; break;
            case GateType::TDG : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::TDG)][1]; break;
            case GateType::RX : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::RX)][1]; break;
            case GateType::RY : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::RY)][1]; break;
            case GateType::RZ : 
                tim_pred += pergate_single_perf[numQubits][int(GateType::RZ)][1]; break;
            default:
                printf("meet wrong gate : %s\n", Gate::get_name(ty).c_str());
                UNREACHABLE()
        }
    }
    return tim_pred / 1000 / 512 + pergate_group_overhead * (1 << numQubits);
}

double Evaluator::perfBLAS(int numQubits, int blasSize) {
    loadParam(numQubits);
    //double bias = (numQubits < 28) ? ((idx_t)1 << (28 - numQubits)) : (1.0 / ((idx_t)1 << (numQubits - 28)));
    return BLAS_perf[numQubits][blasSize] + cutt_cost[numQubits];
}

bool Evaluator::PerGateOrBLAS(const GateGroup* gg_pergate, const GateGroup* gg_blas, int numQubits, int blasSize) {
    double pergate = perfPerGate(numQubits, gg_pergate);
    double blas = perfBLAS(numQubits, blasSize);
    return pergate / (gg_pergate -> gates).size() < blas / (gg_blas -> gates).size();
}