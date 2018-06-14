export INSTANCE_TYPE=p2.xlarge

docker-machine create --driver amazonec2 --amazonec2-region us-east-1 --amazonec2-zone a --amazonec2-instance-type $INSTANCE_TYPE gpu1
