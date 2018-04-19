sudo apt-get install -y openjdk-8-jdk
sudo update-alternatives --config java; sudo update-alternatives --config javac
cd
wget https://cpachecker.sosy-lab.org/CPAchecker-1.7-unix.tar.bz2
tar -xvjf CPAchecker-1.7-unix.tar.bz2

echo 'export PATH=~/CPAchecker-1.7-unix/scripts:$PATH' >> ~/.paths
export PATH=~/CPAchecker-1.7-unix/scripts:$PATH

