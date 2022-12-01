cd
curl -O https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-$(arch).sh
bash ./Anaconda3-2022.10-Linux-$(arch).sh -b -p $HOME/anaconda3
echo 'path ~/anaconda3/bin' >> ~/.paths
source ~/.paths

conda config --system --add channels defaults
conda config --system --add channels conda-forge
