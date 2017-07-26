cd
git clone https://github.com/GustavePate/lycheesync

unset -f pip

cat $OLDPWD/conf.json | sed s/%pword%/`pword`/g | tee ./lycheesync/ressources/conf.json

sudo apt-get install -y python3-dev python3.4-venv libjpeg-dev zlib1g-dev
cd lycheesync
pyvenv-3.4 ./venv3.4
. ./venv3.4/bin/activate
which pip # should give you a path in your newly created ./venv3.4 dir
# if not execute: curl https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py | python
pip install -r requirements.txt

python main.py ~/Fotos /var/www/html/lychee ./ressources/conf.json
