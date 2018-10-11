FROM centos:7

run yum install -y epel-release
run yum update -y

run yum install -y i3 git tmux wget
run yum groupinstall -y "X Window System"
run yum install -y tigervnc-server tigervnc-server-module libXfont pixman xterm

run mkdir ~/.vnc && \
    echo password | vncpasswd -f > ~/.vnc/passwd && \
    chmod 0600 ~/.vnc/passwd && \
    echo '#!/bin/sh' >  ~/.vnc/xstartup && \
    echo 'i3 >/tmp/log 2>/tmp/log'   >> ~/.vnc/xstartup && \
    chmod +x ~/.vnc/xstartup

run mkdir ~/.i3 && \
    echo '# i3 config file (v4)' >  ~/.i3/config && \
    echo 'set $mod Mod1'         >> ~/.i3/config

run mkdir /usr/share/noVNC && \
    cd /usr/share/noVNC && \
    git clone https://github.com/novnc/noVNC.git /usr/share/noVNC && \
    cp /usr/share/noVNC/vnc.html /usr/share/noVNC/index.html

run mkdir -p /usr/share/noVNC/utils/websockify && \
    git clone https://github.com/novnc/websockify.git /usr/share/noVNC/utils/websockify

run dbus-uuidgen > /etc/machine-id

entrypoint ["sh", "-c", "tmux new -d -s novnc 'cd /usr/share/noVNC; ./utils/launch.sh --listen 8080 --vnc localhost:5901'; tmux new -d -s vnc '/usr/bin/vncserver; read'; tmux new -d -s gui 'while true; do DISPLAY=:1 <exec> <args>; sleep 1; done'; sh"]

expose 8080:8080
expose 5901:5901
