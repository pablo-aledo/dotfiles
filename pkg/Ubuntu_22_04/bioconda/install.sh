mkd ~/bioconda
wget https://raw.githubusercontent.com/bioconda/bioconda-common/master/{common,install-and-set-up-conda,configure-conda}.sh
sudo bash install-and-set-up-conda.sh
sudo ln -s /opt/mambaforge/bin/conda /bin/conda
sudo bash configure-conda.sh
sudo conda install conda-build
echo 'export PATH=/opt/mambaforge/bin:"$PATH"' >> ~/.paths
