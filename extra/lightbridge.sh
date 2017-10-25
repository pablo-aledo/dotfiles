wget pabloaledo.ddns.net/default_kp.pem
chmod 600 default_kp.pem
nohup ssh -i default_kp.pem -nNT -R 2222:localhost:22 ubuntu@pabloaledo.ddns.net &
