# UniQ
UniQ is a unified programming model for efficient quantum circuit simulation. It supports state vector simulation, density matrix simulation by gates, and density matrix simulation by Kraus operators on various hardware. UniQ can accelerate quantum circuit simulation by up to 28.59× (4.47× on average) over state-of-the-art frameworks, and successfully scale to 399,360 cores on 1,024 nodes.

## Reproduce Our Results
We have provided dockers on CPU and GPU for reproducing our results. The docker images are available at [https://doi.org/10.5281/zenodo.6628189](https://doi.org/10.5281/zenodo.6628189). The code inside the docker images is available at [https://doi.org/10.5281/zenodo.6628201](https://doi.org/10.5281/zenodo.6628201)

# Get the Source Code
Use the following command to clone UniQ and its submodules:

`git clone https://github.com/thu-pacman/UniQ.git --recursive`

Files in `tests/` are saved with git lfs. Please ensure these files are downloaded.

## Usage of UniQ
UniQ needs to be recompiled (by `./compile.sh -Dxxxx=xxxx` in artifact-evaluation folder) to support different simulation methods and hardware.

To configure UniQ to a specific simulation methods, please use -DMODE option. The available options are:
* statevec: state vector simulation
* densitypure: density matrix simulation by gates
* densityerr: density matrix simulation by Kraus operators

To configure UniQ to a specific hardware, please use -DHARDWARE option. The available options are:
* cpu
* gpu

For the full commands, please refer to artifact-evaluation folder. These scripts are used to generate the results in our paper.
* Fig.8: sv-strong-gpu.sh
* Fig.9: pure-strong-gpu.sh
* Fig.10: pure-strong-gpu-nvprof.sh
* Fig.11: sv-strong-cpu.sh
* Fig.12: err-strong-gpu.sh
* Fig.13 and Fig.14: sv-breakdown-cpu.sh
* Fig.16: dm-cpu.sh

## Citation
If you find UniQ useful for your research, please cite our paper: TODO