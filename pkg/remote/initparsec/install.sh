mouseover 10 10000 parsec.png; xdotool click 1; xdotool click 1; sleep 10
mouseover 10 10000 maximize.png && xdotool click 1;

xdotool key Return
xdotool type $parsec_email
xdotool key Tab
xdotool type $parsec_pw
xdotool key Return

sleep 5
xdotool key Return
sleep 5

mouseover 10 10000 closewin.png && xdotool click 1;
