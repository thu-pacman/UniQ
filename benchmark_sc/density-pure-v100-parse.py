import numpy as np
import pandas
circuits=['bv_13', 'hidden_shift_12', 'ising_9', 'qaoa_10', 'quantum_volume_11', 'supremacy_12', 'basis_change_9', 'basis_change_10', 'basis_change_11', 'basis_change_12', 'basis_change_13', 'basis_change_14']
results = np.zeros((len(circuits), 5))
gates = np.zeros((len(circuits),), dtype=np.int)
for i, c in enumerate(circuits):
    for j, gpu in enumerate([1, 2, 4, 8, 16]):
        with open(f"../build/logs/density_pure_v100/{c}_{gpu}.log") as f:
            for st in f.readlines():
                if "Logger[0]: Time Cost:" in st:
                    results[i][j] = float(st[22:-3]) / 1000.
                elif j == 1 and "Logger[0]: Total Gates" in st:
                    gates[i] = int(st.strip().split()[-1])
df = pandas.DataFrame(results)
print(df.to_string(index=False))
df = pandas.DataFrame(gates)
print(df.to_string(index=False))
# np.set_printoptions(precision=4)
# print(results)