import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import phase_amplitude_damping_error
import os

def round_zero(x):
    if x > -1e-14 and x < 1e-14:
        return 0.0
    else:
        return x


def run(filename):
    file_path = os.path.join(f"/home/heheda/QCSimulator/tests/input-extend/{filename}.qasm")
    circ = QuantumCircuit.from_qasm_file(file_path)
    circ.save_density_matrix()
    num_qubits = circ.num_qubits
    backend = AerSimulator(method='density_matrix')
    backend.set_option('device', 'GPU')
    result = backend.run(circ).result()
    print(f"{filename}: {result.time_taken}", flush=True)
    output = result.results[0].data.density_matrix
    with open(f"output-pure/{filename}.log", "w") as f:
        for idx in range(128):
            prob = output[idx][idx]
            l = prob.real * prob.real + prob.imag * prob.imag
            f.write("{} {} {:.12f}: {:.12f} {:.12f}\n".format(idx, idx, l, round_zero(prob.real), round_zero(prob.imag)))
        for idx1 in range(2 ** num_qubits):
            for idx2 in range(2 ** num_qubits):
                if idx1 == idx2 and idx1 < 128:
                    continue
                prob = output[idx1][idx2]
                l = prob.real * prob.real + prob.imag * prob.imag
                if l > 0.001:
                    f.write("{} {} {:.12f}: {:.12f} {:.12f}\n".format(idx1, idx2, l, round_zero(prob.real), round_zero(prob.imag)))


if __name__ == "__main__":
    cases = [
        "bv_13",
        "efficient_su2_11",
        "hidden_shift_12",
        "iqp_14",
        "qaoa_14",
        "qft_13",
        "supremacy_12"
    ]

    for case in cases:
        run(case)

