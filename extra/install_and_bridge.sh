#!/bin/bash

ROOT=$(dirname $(readlink -f $0))/..
cd $ROOT/extra

# install
../install.sh

# ssh server
source $ROOT/source/.shell/server
server ssh

# install aws
source $ROOT/source/.shell/pkg
pkg install aws

# aws_server
source $ROOT/source/.shell/aws 
aws_config
sed -i 's/AWS_SERVER=.*/AWS_SERVER=pabloaledo.ddns.net/g' /tmp/aws_config
aws_bridge_server
