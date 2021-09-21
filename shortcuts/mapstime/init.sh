export DISPLAY=:1
echo $1 > /tmp/departure
echo $2 > /tmp/arrival
xdotool key 6
