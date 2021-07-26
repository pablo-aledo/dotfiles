mouseover -1 10000 win.png && xdotool click 1
mouseover -1 10000 powershell.png && xdotool click 1
mouseover -1 10000 maximize.png && xdotool click 1
sleep 10

xdotool type '[Net.ServicePointManager]::SecurityProtocol = "tls12, tls11, tls"'; xdotool key Return; sleep 10
xdotool type --delay 200 '(New-Object System.Net.WebClient).DownloadFile("https://github.com/acceleration3/cloudgamestream/archive/refs/heads/master.zip","$ENV:UserProfile\Downloads\acceleration3s.zip")'; xdotool key Return; sleep 10
xdotool type --delay 200 'New-Item -Path $ENV:UserProfile\Downloads\acceleration3s -ItemType Directory'; xdotool key Return; sleep 10
xdotool type --delay 200 'Expand-Archive $ENV:UserProfile\Downloads\acceleration3s.zip -DestinationPath $ENV:UserProfile\Downloads\acceleration3s'; xdotool key Return; sleep 10
xdotool type --delay 200 'CD $ENV:UserProfile\Downloads\acceleration3s\cloudgamestream-master'; xdotool key Return; sleep 10
xdotool type --delay 200 'Set-ExecutionPolicy Unrestricted; ./Setup.ps1'; xdotool key Return; sleep 10

mouseover -1 10000 yes.png && xdotool click 1
