cd /tmp/
wget https://aka.ms/downloadazcopy-v10-linux -O downloadazcopy-v10-linux.tgz
tar -xvzf downloadazcopy-v10-linux.tgz
sudo mv ./azcopy_linux_amd64_10.25.1/azcopy /bin
rm -fr ./azcopy_linux_amd64_10.25.1/
