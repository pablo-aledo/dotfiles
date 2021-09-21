export DISPLAY=:1
echo $1 > /tmp/departure
echo $2 > /tmp/arrival
rm -f /tmp/time
xdotool key 6
