import os
import qiskit
from qiskit import QuantumCircuit
from qiskit import Aer
import time

def run(filename):
    file_path = os.path.join(f"/home/heheda/QCSimulator/tests/input-extend/{filename}.qasm")
    # file_path = os.path.join(f"/home/heheda/QCSimulator/tests/{filename}.qasm")
    simulator = Aer.get_backend('aer_simulator_statevector')
    # print(simulator.options)
    options = {
        "method": "statevector",
        "device": "CPU",
        "precision": "double",
    }
    simulator.set_options(**options)
    circ = QuantumCircuit.from_qasm_file(file_path)
    circ.save_statevector()
    start = time.time()
    result = qiskit.execute(circ, backend=simulator).result()
    end = time.time()
    print(filename, result.time_taken, end - start, flush=True)
    # print(result)

if __name__ == "__main__":
    # cases = [
    #     "basis_change_24",
    #     "bv_24",
    #     "efficient_su2_25",
    #     "hidden_shift_27",
    #     "qft_26",
    #     "supremacy_25"
    # ]
    cases = [
        "bv_27",
        "efficient_su2_28",
        "hidden_shift_27",
        "iqp_25",
        "qaoa_26",
        "qft_29",
        "supremacy_28"
    ]
    # cases = [
    #     "basis_change_24",
    #     "basis_change_25",
    #     "basis_change_26",
    #     "basis_change_27",
    #     "basis_change_28",
    # ]

    for case in cases:
        run(case)

