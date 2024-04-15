wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y blobfuse

#pkg install libfuse2

#cat << EOF > ~/fuse_connection.cfg
#accountName ...
#accountKey ...
#containerName ...
#EOF

#blobfuse ~/mycontainer --tmp-path=~/mnt/ --config-file=fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
