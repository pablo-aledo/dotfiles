source config.sh

mouseover 10 10000 chrome.png && xdotool click 1; xdotool click 1
xdotool type https://builds.parsecgaming.com/package/parsec-windows.exe
xdotool key Return

mouseover 10 10000 parsec.png && xdotool click 1;

xdotool key Alt+y

xdotool type $parsec_user
xdotool key Tab
xdotool type $parsec_pw
xdotool key Return

echo "Confirm email"; read
xdotool key Return
xdotool key Return
xdotool key Return
xdotool key Return

mouseover 10 10000 close.png && xdotool click 1;

mouseover 10 10000 maximize.png && xdotool click 1;
xdotool click 5
mouseover 10 10000 share.png && xdotool click 1;

mouseover 10 10000 closewin.png && xdotool click 1;



xdotool key Ctrl+w
