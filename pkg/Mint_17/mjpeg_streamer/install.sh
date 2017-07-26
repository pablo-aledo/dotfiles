sudo apt-get install -y libjpeg62-dev

cd /tmp/ 
wget https://github.com/jacksonliam/mjpg-streamer/archive/master.zip
#http://downloads.sourceforge.net/project/mjpg-streamer/mjpg-streamer/Sourcecode/mjpg-streamer-r63.tar.gz

unzip master.zip
cd mjpg-streamer-master/mjpg-streamer-experimental

sed -i 's/PLUGINS += input_raspicam.so//g' Makefile
make; sudo make install

echo '#!/bin/bash'                                                                                                                                    | sudo tee /bin/webcam
echo 'while(true)'                                                                                                                                    | sudo tee -a /bin/webcam
echo 'do'                                                                                                                                             | sudo tee -a /bin/webcam
echo 'export LD_LIBRARY_PATH=/usr/local/lib/'                                                                                                         | sudo tee -a /bin/webcam
echo 'mjpg_streamer --input "input_uvc.so --device /dev/video0 --fps 5 --resolution 640x480" --output "output_http.so --port 8080 -w /usr/local/www"' | sudo tee -a /bin/webcam
echo 'sleep 1'                                                                                                                                        | sudo tee -a /bin/webcam
echo 'done'                                                                                                                                           | sudo tee -a /bin/webcam
sudo chmod +x /bin/webcam
sudo webcam
