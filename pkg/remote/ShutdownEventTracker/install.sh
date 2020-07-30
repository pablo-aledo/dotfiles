mouseover 10 10000 win.png && xdotool click 1
mouseover 10 10000 powershell.png && xdotool click 1
mouseover 10 10000 maximize.png && xdotool click 1
xdotool type 'regedit.exe'; sleep 5; xdotool key Return; sleep 10
mouseover -1 10000 yes.png && xdotool click 1; sleep 5

xdotool key Ctrl+f; sleep 1
xdotool type 'DirtyShutdown'; sleep 5;
xdotool key Return; sleep 10

xdotool key Return; sleep 1;
xdotool key 0; sleep 1;
xdotool key Return; sleep 1

for a in $(seq 1 5)
do
    mouseover 1 10000 closewin1.png closewin2.png closewin3.png && xdotool click 1
done
