export CONDA_BASE=~/conda
export CONDA_PATH=~/conda
mkd $CONDA_BASE
curl -s -o conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 700 $CONDA_BASE/conda.sh
$CONDA_BASE/conda.sh -u -b -p $CONDA_PATH
$CONDA_PATH/bin/conda init
. $HOME/.bashrc
pip install  ...
