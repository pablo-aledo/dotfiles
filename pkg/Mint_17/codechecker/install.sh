sudo apt-get install -y clang-3.9 build-essential curl doxygen gcc-multilib git python-virtualenv python-dev thrift-compiler
git clone https://github.com/Ericsson/CodeChecker.git --depth 1 ~/codechecker
cd ~/codechecker
make venv
source $PWD/venv/bin/activate
make package
export PATH="$PWD/build/CodeChecker/bin:$PATH"
cd ..
deactivate

sudo cp ~/.dotfiles/pkg/Mint_17/codechecker/codecheck.sh /usr/bin
sudo chmod +x /usr/bin/codecheck.sh
