mouseover 10 10000 win.png && xdotool click 1
mouseover 10 10000 powershell.png && xdotool click 1
mouseover 10 10000 maximize.png && xdotool click 1
sleep 10

xdotool type '[Net.ServicePointManager]::SecurityProtocol = "tls12, tls11, tls"'; xdotool key Return; sleep 10
xdotool type --delay 200 '(New-Object System.Net.WebClient).DownloadFile("https://github.com/jamesstringerparsec/Parsec-Cloud-Preparation-Tool/archive/master.zip","$ENV:UserProfile\Downloads\Parsec-Cloud-Preparation-Tool.zip")'; xdotool key Return; sleep 10
xdotool type --delay 200 'New-Item -Path $ENV:UserProfile\Downloads\Parsec-Cloud-Preparation-Tool -ItemType Directory'; xdotool key Return; sleep 10
xdotool type --delay 200 'Expand-Archive $ENV:UserProfile\Downloads\Parsec-Cloud-Preparation-Tool.Zip -DestinationPath $ENV:UserProfile\Downloads\Parsec-Cloud-Preparation-Tool'; xdotool key Return; sleep 10
xdotool type --delay 200 'CD $ENV:UserProfile\Downloads\Parsec-Cloud-Preparation-Tool\Parsec-Cloud-Preparation-Tool-master\'; xdotool key Return; sleep 10
xdotool type --delay 200 'Powershell.exe -File $ENV:UserProfile\Downloads\Parsec-Cloud-Preparation-Tool\Parsec-Cloud-Preparation-Tool-master\Loader.ps1'; xdotool key Return; sleep 10

mouseover -1 10000 yes.png && xdotool click 1
mouseover -1 10000 yn.png && xdotool key n; xdotool key Return
mouseover 20 10000 cancel.png && xdotool click 1
mouseover -1 10000 tesla.png
mouseover -1 10000 yn.png && xdotool key n; xdotool key Return
mouseover -1 10000 enter.png && xdotool click 1 && xdotool key Return

for a in $(seq 1 10)
do
    mouseover 1 10000 closewin1.png closewin2.png closewin3.png && xdotool click 1
done
