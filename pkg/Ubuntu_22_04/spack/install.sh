sudo apt update

sudo apt-get install g++-12 gfortran-12

sudo apt install -y build-essential ca-certificates coreutils curl environment-modules gfortran git gpg lsb-release python3 python3-distutils python3-venv unzip zip

cd
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
cd spack
git checkout v0.20.0
. share/spack/setup-env.sh
echo '. ~/spack/share/spack/setup-env.sh' >> ~/.paths

# cd
# mkd spack_yaml
# spack env activate .
# spack concretize -f
# spack install

