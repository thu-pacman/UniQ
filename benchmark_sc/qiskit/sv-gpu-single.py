import os
import qiskit
from qiskit import QuantumCircuit
from qiskit import Aer
import time

def run(filename):
    file_path = os.path.join(f"/home/heheda/QCSimulator/tests/input-extend/{filename}.qasm")
    simulator = Aer.get_backend('aer_simulator_statevector')
    options = {
        "method": "statevector",
        "device": "GPU",
        "precision": "double",
    }
    simulator.set_options(**options)
    circ = QuantumCircuit.from_qasm_file(file_path)
    circ.save_statevector()
    start = time.time()
    result = qiskit.execute(circ, backend=simulator).result()
    end = time.time()
    print(filename, result.time_taken, end - start, flush=True)

if __name__ == "__main__":
    cases = [
        "bv_27",
        "efficient_su2_28",
        "hidden_shift_27",
        "iqp_25",
        "qaoa_26",
        "qft_29",
        "supremacy_28"
    ]

    for case in cases:
        run(case)

