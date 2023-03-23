for y in $(seq 200 50 650)
do
    xdotool mousemove 300 $y; sleep 1
    xdotool click 3; sleep 1
    xdotool key Down; sleep 1
    xdotool key Return; sleep 1
    xdotool mousemove 1500 500; sleep 1
    xdotool click 1; sleep 3
    xdotool type yes; sleep 1
    xdotool key Return; sleep 1
    xdotool key Alt+F4; sleep 1
    xdotool key Escape; sleep 1
done
