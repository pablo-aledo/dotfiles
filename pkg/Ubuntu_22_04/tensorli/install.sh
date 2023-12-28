# create conda env (if you want to do it faster use mamba)
# conda env create -f environment.yml
# (if you want to do it faster use mamba)
mamba env create -f environment.yml
# activate env
conda activate tensorli
# python path need to be set to the root of the project
export PYTHONPATH=$PWD
# run all tests
pytest
# run specific test (verbose)
pytest -v -rP -k "transformer"
