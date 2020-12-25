FROM ubuntu:18.04

RUN apt-get update && apt-get install -y ffmpeg npm xvfb
run mkdir -p /root/.cache/node-gyp/
run npm install --unsafe-perm -g ffmpeg-concat

run echo '#!/bin/bash' >> /bin/ffcat
run echo 'xvfb-run -s "-ac -screen 0 1280x1024x24" ffmpeg-concat --frame-format png -o concat.mp4 $*' >> /bin/ffcat
run chmod +x /bin/ffcat

WORKDIR /workdir
entrypoint [ "ffcat" ]
