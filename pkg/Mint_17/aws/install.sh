sudo apt-get install -y --force-yes python-pip
sudo pip install awscli
mkdir ~/.aws
sudo install -o `whoami` -m 700 $REPOSITORY_FOLDER/Mint17/aws/.aws/config ~/.aws
sudo install -o `whoami` -m 700 $REPOSITORY_FOLDER/Mint17/aws/.aws/credentials ~/.aws
sudo install -o `whoami` -m 400 $REPOSITORY_FOLDER/Mint17/aws/default_kp.pem /tmp/
