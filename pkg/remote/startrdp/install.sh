google-chrome cloud.google.com >/dev/null 2>/dev/null &; sleep 10
xdotool mousemove 1150 130; xdotool click 1; sleep 10
xdotool mousemove 700 580; xdotool click 1; sleep 10

mouseover 10 10000 rdpoff.png; sleep 1
xdotool mousemove_relative 70 0; sleep 1
xdotool click 1; sleep 1
xdotool mousemove_relative 0 30; sleep 1
xdotool click 1; sleep 1
xdotool mousemove 970 510; sleep 1
xdotool click 1; sleep 1

sleep 60

mouseover 100 10000 rdpon.png; sleep 1
xdotool mousemove_relative 30 0; sleep 1
xdotool click 1; sleep 1
xdotool mousemove_relative 0 30; sleep 1
xdotool click 1; sleep 1
xdotool mousemove 820 560; sleep 1
xdotool click 1; sleep 1
mouseover 60 10000 copy.png; sleep 1
xdotool click 1; sleep 1
xdotool mousemove 860 510; sleep 1
xdotool click 1; sleep 1

mouseover 10 10000 rdpon.png; sleep 1
xdotool click 1; sleep 10
xdotool key Ctrl+v; sleep 1
xdotool key Return; sleep 1
