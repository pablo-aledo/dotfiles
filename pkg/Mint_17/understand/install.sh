cd /opt/
wget http://builds.scitools.com/all_builds/b895/Understand/Understand-4.0.895-Linux-64bit.tgz -O - | sudo tar -xvz
echo 'export PATH=/opt/scitools/bin/linux64:$PATH' >> ~/.paths
export PATH=/opt/scitools/bin/linux64/:$PATH
