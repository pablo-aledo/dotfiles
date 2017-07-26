# Folder
cd 
mkdir svcomp
cd svcomp

# BenchExec
sudo apt-get install -y python3-pip git libxml2-dev libxslt1-dev python3-lxml
pip3 install --user git+https://github.com/sosy-lab/benchexec.git
echo 'export PATH=~/.local/bin:$PATH' >> ~/.paths 
export PATH=~/.local/bin:$PATH
sudo mount -t cgroup none /sys/fs/cgroup 
sudo chmod o+wt '/sys/fs/cgroup/'

# Benchmarks
git clone --depth=1 https://github.com/sosy-lab/sv-benchmarks

# Cpachecker
sudo apt-get install -y ant openjdk-7-jdk
sudo update-alternatives --config java
sudo update-alternatives --config javac
git clone --depth=1 https://github.com/sosy-lab/cpachecker
cd cpachecker
ant

# Lamp server
sudo apt-get install -y apache2

