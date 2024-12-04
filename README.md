# Installation 

### Setup spack and add NESO-spack repository 

Clone spack and source spack setup 
``` 
git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ~/spack 
cd ~/spack 
. share/spack/setup-env.sh 
``` 

Enable `tcl` to install module tool like lmod and source the bash file to make commands available 

``` 
spack config add "modules:default:enable:[tcl]" 
spack install lmod 
. $(spack location -i lmod)/lmod/lmod/init/bash 
. spack/share/spack/setup-env.sh 
``` 

Clone NESO-spack and add to spack 

``` 
git clone git@github.com:ExCALIBUR-NEPTUNE/NESO-spack.git 
cd NESO-spack 
spack repo add . 
``` 

### Install prerequisites using spack 

Use spack to install: adaptivecpp, python, mpich, googletest, py-pip, py-setuptools, python-venv and cmake 

Important to note that firedrake must be built with the same version of python as that for adaptivecpp. Also note that firedrake requires vtk to work which might restrict using newer versions of python. The installation has been tested to work with Python 3.11.9. You can constrain the install to specific version of python using command like: 


`spack install neso.adaptivecpp ^python@3.11.9` 

### Install Firedrake
After installing all the prerequisites, load them with a script, making sure the version of python is the same as earlier, like: 

```
module load adaptivecpp/<version-hash> 
module load python/<version-hash> 
module load mpich/<version-hash> 
module load googletest/<version-hash> 
module load py-pip/<version-hash> 
module load py-setuptools/<version-hash> 
module load cmake/<version-hash> 
module load python-venv/<version-hash>
```
Download the firedrake install script and finally install it while pointing it to the relevant MPICH compilers and a virtual environment name of choice with a command similar to:

```
export VENV_NAME=<VENV_NAME>
python firedrake-install --mpicc $(spack location -i mpich@<version>)/bin/mpicc --mpicxx $(spack location -i mpich@<version>)/bin/mpicxx --mpif90 $(spack location -i mpich@<version>)/bin/mpif90 --mpiexec $(spack location -i mpich@<version>)/bin/mpiexec --honour-pythonpath --install irksome --netgen --venv-name $VENV_NAME
```
### Clone and build NESOdrake
Once installed, find and export the locations of hdf5 and pkgconfig within the Firedrake PETSc and activate firedrake environment using:
```
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:<full_path_to_firedrake_virtual_environment>/src/petsc/default/externalpackages/hdf5<version>

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:<full_path_to_firedrake_virtual_environment>/src/petsc/default/lib/pkgconfig 

source <full_path_to_firedrake_virtual_environment>/bin/activate
```
Clone NESOdrake, install pybind11 globally and finally build:

```
pip install "pybind11[global]"
git clone --recursive git@github.com:will-saunders-ukaea/NESODrake-TwoStream.git
cd NESODrake-TwoStream
mkdir build
cd build
cmake ..
make
export PYTHONPATH=<full_path_to_so_file>/build:$PYTHONPATH
```