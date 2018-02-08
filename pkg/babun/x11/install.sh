pact install xorg-server xinit xhost tmux
tmux new -d -s X11 "XWin :0 -listen tcp -multiwindow"
echo "export DISPLAY=:0.0" >> ~/.paths 
export DISPLAY=:0.0

