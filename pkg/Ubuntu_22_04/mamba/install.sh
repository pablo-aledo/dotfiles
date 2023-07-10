cd
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash "./Mambaforge-$(uname)-$(uname -m).sh" -b -p $HOME/mamba
echo 'path ~/mamba/bin' >> ~/.paths
source ~/.paths

cp $OLDPWD/mambarc .mambarc
