mouseover 10 10000 win.png && xdotool click 1
mouseover 10 10000 powershell.png && xdotool click 1
sleep 1

cat ../../babun/parsec/script.ps1 | while read line
do
    xdotool type "$line"
    xdotool key Return
    sleep 1
done
