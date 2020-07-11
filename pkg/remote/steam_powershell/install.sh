mouseover 10 10000 win.png && xdotool click 1
mouseover 10 10000 powershell.png && xdotool click 1
mouseover 10 10000 maximize.png && xdotool click 1
sleep 10

xdotool type --delay 200 '(New-Object System.Net.WebClient).DownloadFile("https://steamcdn-a.akamaihd.net/client/installer/SteamSetup.exe","$ENV:UserProfile\Downloads\SteamSetup.exe")'; xdotool key Return; sleep 10
xdotool type --delay 200 'Invoke-Item $ENV:UserProfile\Downloads\SteamSetup.exe'; xdotool key Return; sleep 10

mouseover -1 10000 yes.png && xdotool click 1
mouseover -1 10000 next.png && xdotool click 1
mouseover -1 10000 next.png && xdotool click 1
mouseover -1 10000 install.png && xdotool click 1
mouseover -1 10000 checkbox.png && xdotool click 1
mouseover -1 10000 finish.png && xdotool click 1

for a in $(seq 1 10)
do
    mouseover 1 10000 closewin1.png closewin2.png && xdotool click 1
done
