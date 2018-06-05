#!/bin/zsh

source ~/codechecker/venv/bin/activate

# Path of CodeChecker package
# NOTE: SKIP this line if you want to always specify CodeChecker's full path.
export PATH=~/codechecker/build/CodeChecker/bin:$PATH

# Path of `scan-build.py` (intercept-build)
# NOTE: SKIP this line if you don't want to use intercept-build.
#export PATH=~/<user path>/llvm/tools/clang/tools/scan-build-py/bin:$PATH
export PATH=/usr/share/clang/scan-build-py-3.9/bin:$PATH

# Path of the built LLVM/Clang
# NOTE: SKIP this line if clang is available in your PATH as an installed Linux package.
#export PATH=~/<user path>/build/bin:$PATH

CodeChecker check -b "cd ~/codechecker/tests/projects/single_bug/ && make clean && make" -o ~/results
CodeChecker server &
CodeChecker store ~/results -n my-project
google-chrome localhost:8001

