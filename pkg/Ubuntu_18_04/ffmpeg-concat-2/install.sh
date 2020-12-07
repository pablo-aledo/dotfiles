sudo apt-get install -y ffmpeg npm xvfb
sudo mkdir -p /root/.cache/node-gyp/
sudo npm install --unsafe-perm -g ffmpeg-concat
# xvfb-run -s "-ac -screen 0 1280x1024x24" ffmpeg-concat --frame-format jpg -o concat.mp4 input_1.mp4 input_2.mp4
