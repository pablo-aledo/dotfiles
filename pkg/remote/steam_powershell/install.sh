mouseover 10 10000 win.png && xdotool click 1
mouseover 10 10000 powershell.png && xdotool click 1
mouseover 10 10000 maximize.png && xdotool click 1
sleep 10

xdotool type --delay 200 '(New-Object System.Net.WebClient).DownloadFile("https://steamcdn-a.akamaihd.net/client/installer/SteamSetup.exe","$ENV:UserProfile\Downloads\SteamSetup.exe")'
xdotool type --delay 200 '$ENV:UserProfile\Downloads\SteamSetup.exe'
