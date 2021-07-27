export DISPLAY=:1
xrandr --output VNC-0 --mode 1280x800

google-chrome --profile-directory=cloud console.cloud.google.com/compute/instances >/dev/null 2>/dev/null &; sleep 10

mouseover -1 10000 rdpoff.png && xdotool mousemove 0 0 && sleep 1
mouseover -1 10000 rdpoff.png; sleep 1
xdotool mousemove_relative 80 0; sleep 1
xdotool click 1; sleep 1
xdotool mousemove_relative 0 30; sleep 1
xdotool click 1; sleep 1
mouseover -1 10000 start.png; sleep 1
xdotool click 1; sleep 1

mouseover -1 10000 rdpon.png && xdotool click 1
sleep 30

mouseover -1 10000 ok.png && xdotool click 1
mouseover -1 10000 password.png && xdotool mousemove_relative 100 0 && xdotool click 1 && sleep 1 && xdotool type $(pword rdp) && xdotool key Return

mouseover 20 10000 continue.png && xdotool click 1
mouseover 20 10000 cancel.png && xdotool click 1

xdotool key Alt+f
sleep 1
xdotool key Alt
