# FerroX
FerroX is a massively parallel, 3D phase-field simulation framework for modeling ferroelectric materials based scalable logic devices. We  self-consistently solve the time-dependent Ginzburg Landau (TDGL) equation for ferroelectric polarization, Poisson's equation for electric potential, and semiconductor charge equation for carrier densities in semiconductor regions. The algorithm is implemented using Exascale Computing Project software framework, AMReX, which provides effective scalability on manycore and GPU-based supercomputing architectures. The code can be used for simulations of ferroelectric domain-wall induced negative capacitance (NC) effect in Metal-Ferroelectric-Insulator-Metal (MFIM) and Metal-Ferroelectric-Insulator-Semiconductor-Metal (MFISM) devices.

For questions, please reach out to Zhi (Jackie) Yao (jackie_zhiyao@lbl.gov) and Andy Nonaka (ajnonaka@lbl.gov).

# Getting Help
Our community is here to help. Please report installation problems or general questions about the code in the github [Issues](https://github.com/AMReX-Microelectronics/FerroX/issues) tab above.
# Installation
## Download AMReX Repository
``` git clone git@github.com:AMReX-Codes/amrex.git ```
## Download FerroX Repository
``` git clone git@github.com:AMReX-Microelectronics/FerroX.git ```
## Build

### GNU Make (Primary)
Make sure that the AMReX and FerroX are cloned in the same location in their filesystem. Navigate to the Exec folder of FerroX and execute
```make -j 4``` for a GPU build and ```make -j 4 USE_CUDA=FALSE``` for a CPU build.

### CMake (Alternative)
FerroX also supports building with CMake, which automatically downloads and builds dependencies.

#### Basic CMake Build
```bash
# CPU build with default options (OpenMP)
cmake -S . -B build
cmake --build build -j 4

# GPU build (CUDA)
cmake -S . -B build -DFerroX_COMPUTE=CUDA
cmake --build build -j 4
```

#### Core Configuration Options
- **FerroX_COMPUTE**: `NOACC`, `OMP` (default), `CUDA`, `SYCL`, `HIP` - Computing backend
- **FerroX_PRECISION**: `SINGLE`, `DOUBLE` (default) - Floating point precision
- **FerroX_EB**: `OFF` (default), `ON` - Embedded boundary support
- **FerroX_TIME_DEPENDENT**: `OFF` (default), `ON` - Time-dependent simulations
- **FerroX_SUNDIALS**: `OFF` (default), `ON` - SUNDIALS ODE solver support
- **FerroX_MPI**: `ON` (default), `OFF` - Multi-node support
- **FerroX_SIMD**: `OFF` (default), `ON` - CPU SIMD acceleration

#### Print Debug Options (matching GNU Make)
- **FerroX_PRINT_HIGH**: `OFF` (default), `ON` - High level debug printing
- **FerroX_PRINT_MEDIUM**: `OFF` (default), `ON` - Medium level debug printing
- **FerroX_PRINT_LOW**: `OFF` (default), `ON` - Low level debug printing
- **FerroX_PRINT_NAME**: `OFF` (default), `ON` - Function name debug printing

#### External Dependencies

**AMReX Configuration:**
```bash
# Use external AMReX installation
cmake -S . -B build \
  -DFerroX_amrex_internal=OFF \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX

# Use local AMReX source directory
cmake -S . -B build -DFerroX_amrex_src=/path/to/amrex/source

# Use custom AMReX repository/branch
cmake -S . -B build \
  -DFerroX_amrex_repo=https://github.com/user/amrex.git \
  -DFerroX_amrex_branch=my_branch

# Test with specific AMReX pull request (CI/testing)
cmake -S . -B build -DFerroX_amrex_pr=1234
```

**SUNDIALS Configuration (when FerroX_SUNDIALS=ON):**
```bash
# Use external SUNDIALS installation
cmake -S . -B build \
  -DFerroX_SUNDIALS=ON \
  -DFerroX_sundials_internal=OFF \
  -DSUNDIALS_DIR=/path/to/sundials/lib/cmake/sundials

# Use local SUNDIALS source directory
cmake -S . -B build \
  -DFerroX_SUNDIALS=ON \
  -DFerroX_sundials_src=/path/to/sundials/source
```

#### Advanced Build Examples
```bash
# CPU build with embedded boundaries and time-dependent support
cmake -S . -B build \
  -DFerroX_COMPUTE=OMP \
  -DFerroX_EB=ON \
  -DFerroX_TIME_DEPENDENT=ON

# GPU build with SUNDIALS support
cmake -S . -B build \
  -DFerroX_COMPUTE=CUDA \
  -DFerroX_SUNDIALS=ON

# Debug build with all print options enabled
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DFerroX_PRINT_HIGH=ON

# Build with external AMReX using CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=/path/to/amrex/install:$CMAKE_PREFIX_PATH
cmake -S . -B build -DFerroX_amrex_internal=OFF
```

# Running FerroX
Example input scripts are located in `Examples` directory. 

## Simple Testcase
You can run the following to simulate a MFIM heterostructure with a 5 nm HZO as the ferroelectric layer and 4 nm alumina as the dielectric layer under zero applied voltage:

### GNU Make builds (from Exec directory)
```bash
# For MPI+OMP build
mpirun -n 4 ./main3d.gnu.TPROF.MPI.OMP.ex Exec/Examples/inputs_mfim_Noeb

# For MPI+CUDA build
mpirun -n 4 ./main3d.gnu.TPROF.MPI.CUDA.ex Exec/Examples/inputs_mfim_Noeb

# With embedded boundaries
mpirun -n 4 ./main3d.gnu.TPROF.MPI.OMP.EB.ex Exec/Examples/inputs_mfim_eb
```

### CMake builds (from project root directory)
```bash
# For MPI+OMP build
mpirun -n 4 ./build/bin/main3d.gnu.TPROF.MPI.OMP.ex Exec/Examples/inputs_mfim_Noeb

# For MPI+CUDA build  
mpirun -n 4 ./build/bin/main3d.gnu.TPROF.MPI.CUDA.ex Exec/Examples/inputs_mfim_Noeb

# With embedded boundaries (if built with -DFerroX_EB=ON)
export OMP_NUM_THREADS=1; mpirun -n 4 ./build/bin/main3d.gnu.TPROF.MTMPI.OMP.EB.ex Exec/Examples/inputs_mfim_eb

# With time-dependent simulations (if built with -DFerroX_TIME_DEPENDENT=ON)
export OMP_NUM_THREADS=1; mpirun -n 4 ./build/bin/main3d.gnu.TPROF.MTMPI.OMP.TD.ex Exec/Examples/inputs_mfim_Noeb
```
# Visualization and Data Analysis
Refer to the following link for several visualization tools that can be used for AMReX plotfiles. 

[Visualization](https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html)

### Data Analysis in Python using yt 
You can extract the data in numpy array format using yt (you can refer to this for installation and usage of [yt](https://yt-project.org/). After you have installed yt, you can do something as follows, for example, to get variable 'Pz' (z-component of polarization)
```
import yt
ds = yt.load('./plt00001000/') # for data at time step 1000
ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
P_array = ad0['Pz'].to_ndarray()
```
# Publications
1. P. Kumar, M. Hoffmann, A. Nonaka, S. Salahuddin, and Z. Yao, 3D ferroelectric phase field simulations of polycrystalline multi-phase hafnia and zirconia based ultra-thin films, submitted for publication. [arxiv](https://arxiv.org/abs/2402.05331)
2. P. Kumar, A. Nonaka, R. Jambunathan, G. Pahwa, S. Salahuddin, and Z. Yao, FerroX: A GPU-accelerated, 3D Phase-Field Simulation Framework for Modeling Ferroelectric Devices, Computer Physics Communications, 108757, 2023. [link](https://www.sciencedirect.com/science/article/pii/S0010465523001029)
