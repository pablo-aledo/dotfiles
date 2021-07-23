    echo moonligt > /tmp/name
    echo 47984, 47989, 48010 > /tmp/tcp
    echo 5353, 47998, 47999, 48000, 48002, 48010 > /tmp/udp



google-chrome --profile-directory=cloud console.cloud.google.com/networking/firewalls/ >/dev/null 2>/dev/null & sleep 10

mouseover -1 10000 plus.png && xdotool click 1; sleep 1

for n in $(seq 1 3); do xdotool key Tab; done
xdotool type $(cat /tmp/name)

for n in $(seq 1 16); do xdotool key Tab; done
xdotool type $(cat /tmp/name)

for n in $(seq 1 3); do xdotool key Tab; done
xdotool type 0.0.0.0/0

for n in $(seq 1 6); do xdotool key Tab; done; xdotool key space; xdotool key Tab
xdotool type $(cat /tmp/tcp)

for n in $(seq 1 1); do xdotool key Tab; done; xdotool key space; xdotool key Tab
xdotool type $(cat /tmp/udp)

for n in $(seq 1 3); do xdotool key Tab; done;
xdotool key space

