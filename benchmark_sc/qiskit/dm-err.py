import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import phase_amplitude_damping_error
import os

def get_noise_model(n_qubits):
    param_amps = np.array([
        0.13522296, 0.34196305, 0.24942207, 0.20366025, 0.36708856,
        0.22573069, 0.16126925, 0.22023124, 0.19477643, 0.22062259,
        0.34242915, 0.29556578, 0.14447562, 0.24413769, 0.36841306,
        0.29977425, 0.18354474, 0.17749279, 0.35026603, 0.34237515,
        0.30820619, 0.17611002, 0.23364228, 0.20900146, 0.26849671,
        0.19429553, 0.29175968, 0.32635873, 0.20648301, 0.19582834,
        0.16577554, 0.20725059, 0.3529493 , 0.15643779, 0.13911531,
        0.13506932, 0.22451938, 0.19976538, 0.12964262, 0.34413908,
        0.35384347, 0.37235135, 0.34113041, 0.17087591, 0.28486187,
        0.35908144, 0.30639709, 0.30138282, 0.37030199, 0.12811117
    ])
    param_phases = np.array([
        0.27165516, 0.35525184, 0.1916562 , 0.20513042, 0.27364267,
        0.15848727, 0.37182112, 0.30637188, 0.31124254, 0.33848456,
        0.26229897, 0.12982723, 0.32468533, 0.20456679, 0.15046644,
        0.31481037, 0.33237344, 0.22990046, 0.24478173, 0.34522711,
        0.34800876, 0.27030219, 0.14428052, 0.24037756, 0.36350212,
        0.22666077, 0.27186536, 0.16700415, 0.21254885, 0.34969858,
        0.29483833, 0.25706624, 0.27592144, 0.33215269, 0.33985181,
        0.15013914, 0.27628303, 0.2027231 , 0.31656706, 0.27485518,
        0.30443711, 0.3564536 , 0.29340223, 0.19076045, 0.20382232,
        0.15499888, 0.31420134, 0.21966027, 0.24792838, 0.29566892
    ])
    noise_model = NoiseModel()

    for i in range(n_qubits):
        noise_model.add_quantum_error(
            phase_amplitude_damping_error(param_amps[i], param_phases[i]),
            ['u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
            'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 't', 'tdg'],
            [i]
        )
        
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                err_i = phase_amplitude_damping_error(param_amps[i], param_phases[i])
                err_j = phase_amplitude_damping_error(param_amps[j], param_phases[j])
                # err_ij = err_i.tensor(err_j)
                err_ij = err_j.tensor(err_i)
                noise_model.add_quantum_error(
                    err_ij,
                    ['swap', 'cx', 'cy', 'cz', 'csx', 'cp', 'cu', 'cu1', 'cu2', 'cu3', 'rxx',
            'ryy', 'rzz', 'rzx'],
                    [i, j]
                )
    return noise_model


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
    noise_model = get_noise_model(num_qubits)
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    backend.set_option('device', 'GPU')
    result = backend.run(circ).result()
    print(f"{filename}: {result.time_taken}")
    output = result.results[0].data.density_matrix
    with open(f"output/{filename}.log", "w") as f:
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

