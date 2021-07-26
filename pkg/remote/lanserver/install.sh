mouseover -1 10000 win.png && xdotool click 1
mouseover -1 10000 server_manager.png && xdotool click 1
mouseover -1 10000 roles.png && xdotool click 1

xdotool mousemove 0 0; mouseover -1 10000 next.png && xdotool click 1
xdotool mousemove 0 0; mouseover -1 10000 next.png && xdotool click 1
xdotool mousemove 0 0; mouseover -1 10000 next.png && xdotool click 1
xdotool mousemove 0 0; mouseover -1 10000 next.png && xdotool click 1

mouseover -1 1000 down.png && for a in $(seq 1 10); do xdotool click 1; sleep 0.1; done
mouseover -1 10000 lan.png && xdotool click 1
xdotool key space
xdotool mousemove 0 0; mouseover -1 10000 next.png && xdotool click 1
mouseover -1 10000 install.png && xdotool click 1

xdotool mousemove 0 0; mouseover -1 10000 close.png && xdotool click 1
