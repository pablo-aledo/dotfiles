export DISPLAY=:1
xrandr --output VNC-0 --mode 1280x800

google-chrome --profile-directory=cloud cloud.google.com >/dev/null 2>/dev/null &; sleep 10

mouseover 20 10000 console.png && xdotool mousemove 0 0 && sleep 5
mouseover 20 10000 console.png && xdotool click 1; sleep 1

mouseover 20 10000 computeengine.png && xdotool mousemove 0 0 && sleep 1
mouseover 20 10000 computeengine.png && xdotool click 1; sleep 1

mouseover 20 10000 rdpoff.png && xdotool mousemove 0 0 && sleep 1
mouseover 20 10000 rdpoff.png; sleep 1
xdotool mousemove_relative 70 0; sleep 1
xdotool click 1; sleep 1
xdotool mousemove_relative 0 30; sleep 1
xdotool click 1; sleep 1
xdotool mousemove 970 510; sleep 1
xdotool click 1; sleep 1

mouseover 20 10000 rdpon.png; sleep 30

ok=false
while [ $ok = false ]
do
    mouseover 20 10000 rdpon.png; sleep 1
    xdotool mousemove_relative 30 0; sleep 1
    xdotool click 1; sleep 1
    xdotool mousemove_relative 0 30; sleep 1
    xdotool click 1; sleep 1
    xdotool mousemove 820 560; sleep 1
    xdotool click 1; sleep 1
    mouseover 20 10000 copy.png error.png; sleep 1
    if `mouseover 1 10000 error.png`
    then
        ok=false
        xdotool mousemove 730 600; xdotool click 1; sleep 1
    else
        ok=true
    fi
done

xdotool click 1; sleep 1
xdotool mousemove 860 510; sleep 1
xdotool click 1; sleep 1

mouseover 20 10000 rdpon.png; sleep 1
xdotool click 1; sleep 10
xdotool key Ctrl+v; sleep 1
xdotool key Return; sleep 1

mouseover 20 10000 continue.png && xdotool click 1
mouseover 20 10000 cancel.png && xdotool click 1

xdotool key Alt+f
sleep 1
xdotool key Alt
