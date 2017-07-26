wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/pkg          -O - >  .aws
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/easypasswd   -O - >> .aws
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/myshell      -O - >> .aws
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/server       -O - >> .aws
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/aws          -O - >> .aws

echo 'wget pabloaledo.ddns.net/aws_config     -O /tmp/aws_config'                                  >> .aws
echo 'wget pabloaledo.ddns.net/default_kp.pem -O /tmp/default_kp.pem'                              >> .aws

echo 'server ssh'                                                                                  >> .aws
echo 'aws_bridge_server'                                                                           >> .aws

bash .aws
