[ -e ~/Dotfiles/remote/env.sh ] && source ~/Dotfiles/remote/env.sh

mouseover 10 10000 chrome.png && xdotool click 1; xdotool click 1; sleep 10
xdotool type https://steamcdn-a.akamaihd.net/client/installer/SteamSetup.exe
xdotool key Return; sleep 10

mouseover 10 10000 steam.png && xdotool click 1; sleep 10

xdotool key Alt+y

xdotool key Return; sleep 5
xdotool key Return; sleep 5
xdotool key Return; sleep 5
xdotool key Return; sleep 5

mouseover 10 10000 steambar.png && xdotool click 1;

xdotool key Tab
xdotool key Return

xdotool type $steam_user
xdotool key Tab
xdotool type $steam_pw
xdotool key Tab
xdotool key space
xdotool key Tab
xdotool key Return

echo "email code: "; read code
xdotool key Return
xdotool type $code
xdotool key Return
xdotool key Return

xdotool key Ctrl+w

mouseover 10 10000 close.png && xdotool click 1;
