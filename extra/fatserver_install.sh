#!/bin/bash

ROOT=$(dirname $(readlink -f $0))/..
cd $ROOT/extra

../install.sh

# ssh server
cd /media/DATA/Work/dotfiles/data/repository/Mint_17/openssh_server 
source ./install.sh
source $ROOT/source/.shell/easypasswd
easypasswd

# x2go_server
cd /media/DATA/Work/dotfiles/data/repository/Mint_17/x2go_server 
source ./install.sh
echo '#!/bin/sh'               | sudo tee    /bin/starti3
echo 'export LANG=en_US.UTF-8' | sudo tee -a /bin/starti3
echo 'exec i3'                 | sudo tee -a /bin/starti3
sudo chmod +x /bin/starti3

#adapt if I'm using a chromebook as client
source $ROOT/source/.shell/adapt 
adapt_chromebook
