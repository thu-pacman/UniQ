#pragma once

#include "utils.h"
#include "hptt.h"
#include <vector>

namespace CpuImpl {
void initCpu();
void initState(std::vector<cpx*> &deviceStateVec, int numQubits);
void initHpttPlans(std::vector<std::shared_ptr<hptt::Transpose<cpx>>*>& transPlanPointers, const std::vector<int*>& transPermPointers, const std::vector<int>& locals, int numLocalQubits);
void copyBackState(std::vector<cpx>& result, const std::vector<cpx*>& deviceStateVec, int numQubits);
void destroyState(std::vector<cpx*>& deviceStateVec);
}
